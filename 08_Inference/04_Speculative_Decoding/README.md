# 04 — Speculative Decoding

## 1. The Latency Problem — Why Decoding is Slow

```
AUTOREGRESSIVE GENERATION CONSTRAINT:
  Token t can only be generated AFTER tokens 1, ..., t-1 are known.
  
  → Sequential dependency: cannot parallelise across token positions!
  
  LATENCY = T × (time per step)
  
  Time per step for LLaMA-3 8B at batch=1:
    KV cache read:   128 KB × context / (2 TB/s bandwidth) ≈ 0.26 ms (ctx=2048)
    Compute:         2 × 8B × 1 = 16 GFLOPs / (312 TFLOPS) ≈ 0.05 ms
    Total: ~0.3 ms per token  →  about 50 tokens/second

    For T=500 tokens: latency ≈ 10 seconds  (frustratingly slow for interactive use)

HARDWARE BOUND ANALYSIS:
  At batch=1, compute intensity = 2×8B FLOPs / (2 × 8B × 2 bytes for weight read)
                                 = 1 FLOP/byte  ← MUCH less than A100's 312/2000 = 156 FLOP/byte
  
  → The GPU is ~99% idle (waiting for memory, not computing)!
  → Speculative decoding exploits this idle compute capacity.
```

## 2. Speculative Decoding — Exact Algorithm

```
SETUP:
  π_p = target model (large, expensive, LLaMA-3 70B)  p = "primary"
  π_q = draft model  (small, cheap,    LLaMA-3 8B)   q = "quick"
  γ = number of speculative tokens per step

ALGORITHM (one speculative step):

  DRAFT PHASE (serial, using π_q):
    For l = 1 to γ:
      x̃_{t+l} ~ π_q(· | x₁,...,x_{t+l-1})   ← sample from draft model

    Result: draft tokens x̃_{t+1}, ..., x̃_{t+γ}

  VERIFICATION PHASE (parallel, using π_p):
    Compute in ONE forward pass of π_p:
      p_{t+1} = π_p(· | x₁,...,xₜ)
      p_{t+2} = π_p(· | x₁,...,xₜ, x̃_{t+1})
      ...
      p_{t+γ} = π_p(· | x₁,...,xₜ, x̃_{t+1}, ..., x̃_{t+γ-1})
    
    This is ONE forward pass through π_p with γ tokens!
    All γ probabilities computed simultaneously (batch of 1 with multiple queries).

  REJECTION SAMPLING (accept/reject each draft token):
    For l = 1 to γ:
      Let q = π_q(x̃_{t+l} | context)   (draft probability of this token)
      Let p = π_p(x̃_{t+l} | context)   (target probability of this token)
      
      Accept x̃_{t+l} with probability min(1, p/q)
      If rejected: sample new token from adjusted distribution
                   p'(x) = norm(max(0, p(x) − q(x)))   ("correction distribution")
                   x_{t+l} ~ p'
                   STOP (don't accept any more draft tokens in this step)

  OUTPUT: all accepted tokens (and one corrected token if any rejection)
```

## 3. Correctness Proof — Distribution Matching

```
THEOREM: Speculative decoding produces samples IDENTICAL in distribution to the target π_p.

PROOF for a single position (simplify to one draft token):
  x̃ ~ q(x) (draft distribution)
  Accept with probability α = min(1, p(x̃)/q(x̃))

CASE 1: p(x) ≥ q(x) for a particular x:
  P(x is accepted) = q(x) × min(1, p(x)/q(x)) = q(x) × 1 = q(x)  [since p≥q, accept always]

CASE 2: p(x) < q(x):
  P(x is accepted) = q(x) × min(1, p(x)/q(x)) = q(x) × p(x)/q(x) = p(x)

Wait, let's redo: we need the OUTPUT distribution = p(x).
  
P(output = x):
  = P(x is accepted from draft)  +  P(some rejection AND then x is sampled from p')
  = q(x) × min(1, p(x)/q(x))    +  P(rejection occurs) × p'(x)
  
Let Z = ∑_x max(0, p(x)−q(x))  (normalisation constant of p')
P(rejection) = ∑_x q(x) × max(0, 1 − p(x)/q(x)) = ∑_x max(0, q(x)−p(x)) = 1 − ∑_x min(p(x),q(x))

P(output = x) = min(p(x), q(x)) + [1 − ∑_y min(p(y),q(y))] × max(0, p(x)−q(x)) / Z

Let T = ∑_y min(p(y),q(y)).  Then Z = ∑_y max(0, p(y)−q(y)) = 1−T (since ∑p=∑q=1 implies ∑max(0,p-q) = 1−T).

P(output = x) = min(p(x),q(x)) + (1−T) × max(0,p(x)−q(x))/(1−T)
              = min(p(x),q(x)) + max(0, p(x)−q(x))
              = p(x)   [since min(a,b)+max(0,a−b) = a for all a,b] ■

Output distribution = p(x) = target distribution  ✓
Speculative decoding is EXACTLY correct, not approximate!
```

## 4. Expected Speedup — Mathematical Derivation

```
Let α = E[acceptance rate per token] = E[min(1, p(x)/q(x))]
  where the expectation is over x ~ q(x).

ACCEPTANCE RATE FORMULA:
  α = ∑_x q(x) × min(1, p(x)/q(x)) = ∑_x min(p(x), q(x))
    = 1 − (1/2) × TV(p, q)  × 2 = 1 − TV(p, q)
    
  where TV(p,q) = (1/2) ∑_x |p(x)−q(x)|  is the total variation distance.

  α = 1 − TV(p,q) ∈ [0,1]
  α ≈ 1 when p ≈ q (draft ≈ target)  → most tokens accepted!
  α ≈ 0 when p ≠ q completely         → all rejected (no speedup)

EXPECTED ACCEPTED TOKENS PER SPECULATIVE STEP:
  E[accepted tokens] = ∑_{k=1}^{γ} P(first k tokens all accepted)
  
  Assuming i.i.d. acceptance (approximation): P(k accepted) = α^k
  E[accepted] = ∑_{k=1}^{γ} α^k = α(1−α^γ)/(1−α)   (geometric series)
  
  Plus 1 extra token from correction sampling (always generated):
  E[tokens per step] = 1 + α(1−α^γ)/(1−α)

TIME PER SPECULATIVE STEP:
  Drafting (γ serial steps of π_q):  γ × t_q
  Verification (1 parallel step of π_p with γ+1 tokens): t_p × 1 (parallel!)
  Total: γ × t_q + t_p

WITHOUT SPECULATIVE DECODING:
  To generate E[tokens per step] tokens using π_p alone:
  Time = E[tokens per step] × t_p

SPEEDUP:
  S = E[tokens per step] × t_p / (γ × t_q + t_p)
    = [1 + α(1−α^γ)/(1−α)] × t_p / (γ × t_q + t_p)

EXAMPLE:
  α = 0.75,  γ = 4,  t_q = 0.1 ms,  t_p = 1.0 ms (10× larger model)
  
  E[tokens/step] = 1 + 0.75(1−0.75⁴)/(1−0.75) = 1 + 0.75×0.684/0.25 = 1 + 2.05 = 3.05
  
  Time: 4×0.1 + 1.0 = 1.4 ms
  vs naive: 3.05 × 1.0 = 3.05 ms
  
  Speedup = 3.05 / 1.4 = 2.2×
  
  Practical speedups: 2-3× for well-matched draft/target pairs.
```

## 5. Choosing the Draft Model

```
OPTIMAL DRAFT MODEL CRITERIA:
  1. Fast: t_q << t_p  (draft should take <20% of target model time)
  2. Accurate: TV(p, q) small → α high
  3. Same vocabulary as target (required for token matching)

PRACTICAL CHOICES:
  Target: LLaMA-3 70B  → Draft: LLaMA-3 8B   (9× faster, same tokeniser)
  Target: LLaMA-3 8B   → Draft: LLaMA-3 1B   (8× faster, same tokeniser)
  Target: Claude-3 Opus → Draft: Claude-3 Haiku (10× faster, same tokeniser)

SELF-SPECULATIVE DECODING (single model):
  Use early exit at intermediate layers as the "draft":
    Draft: stop at layer L/2, project to vocabulary → approximate logits
    Verify: full forward pass, check if early-exit prediction matches full-pass
  
  No separate draft model needed! Same KV cache reused.
  
  ACCEPTANCE RATE ANALYSIS:
    For each layer l (out of L total):
    α_l ≈ exp(−ε_l)  where ε_l = KL(π_full ‖ π_layer_l)
    
    Layers near the end (l close to L): ε_l ≈ 0 → α_l ≈ 1 (almost always agree)
    Earlier layers: ε_l > 0 → lower acceptance rate
    
    Empirically: using 50% of layers as draft gives α ≈ 0.6-0.8 on many tasks.
```

---

## Exercises

1. Prove algebraically that min(p(x),q(x)) + max(0, p(x)−q(x)) = p(x) for all p(x),q(x)≥0. This is the key identity in the correctness proof.
2. For α=0.8 and γ=5, compute the expected number of accepted tokens per step. What is the speedup if t_q = 0.05ms and t_p = 0.5ms?
3. Show that when p=q (draft perfectly matches target), α=1 and all γ draft tokens are always accepted. What does the speedup formula reduce to in this case?
