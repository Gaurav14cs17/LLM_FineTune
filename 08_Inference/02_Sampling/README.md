# 02 — Sampling Methods

## 1. Temperature Sampling — Mathematical Analysis

```
STANDARD SOFTMAX:
  P(wₜ = k | context) = exp(zₖ) / ∑_j exp(zⱼ)

TEMPERATURE T MODIFICATION:
  P_T(wₜ = k | context) = exp(zₖ/T) / ∑_j exp(zⱼ/T)

  This is equivalent to SCALING ALL LOGITS by 1/T before softmax.

EFFECT ON DISTRIBUTION:
  T → 0 (very cold): all probability concentrates on argmax → greedy decoding
  T = 1 (neutral):   standard model distribution
  T → ∞ (very hot):  all probabilities → 1/|V| → uniform (completely random)

MATHEMATICAL PROOF of T→0 behavior:
  P_T(k) = exp(zₖ/T) / ∑_j exp(zⱼ/T)
  
  Let k* = argmax_k zₖ (the highest-logit token).
  Divide numerator and denominator by exp(z_{k*}/T):
  
  P_T(k) = exp((zₖ − z_{k*})/T) / ∑_j exp((zⱼ − z_{k*})/T)
  
  As T → 0:
    For k = k*: numerator = exp(0) = 1,  denominator → 1 + 0 + 0 + ... = 1
    → P_T(k*) → 1  ✓
    
    For k ≠ k*: numerator = exp((zₖ − z_{k*})/T) → exp(−∞) = 0  (since zₖ < z_{k*})
    → P_T(k) → 0  ✓

TEMPERATURE AND ENTROPY:
  H_T = −∑_k P_T(k) log P_T(k)   (entropy of temperature-scaled distribution)
  
  dH_T/dT > 0:  higher temperature → higher entropy → more diverse outputs
  dH_T/dT < 0:  lower temperature → lower entropy → less diverse, more focused
  
  Optimal T for a task:
    Factual QA: T ≈ 0.1-0.3  (want near-deterministic correct answer)
    Creative writing: T ≈ 0.7-1.2  (want diverse, non-repetitive text)
    Code generation: T ≈ 0.2-0.4  (want syntactically correct output)
```

## 2. Top-k Sampling — Mathematical Definition

```
TOP-k SAMPLING ALGORITHM:
  
  STEP 1: Get logits z ∈ ℝ^{|V|}
  STEP 2: Find top-k tokens: V_k = argmax^k_j z_j  (k tokens with highest logits)
  STEP 3: Zero out all other tokens:
            z̃_j = z_j  if j ∈ V_k
            z̃_j = −∞  otherwise
  STEP 4: Sample from softmax(z̃)

EFFECTIVE DISTRIBUTION:
  P_topk(j) = exp(z_j) / ∑_{i ∈ V_k} exp(z_i)   for j ∈ V_k
  P_topk(j) = 0                                    for j ∉ V_k

PROBLEM WITH TOP-k:
  k is FIXED regardless of how peaked or flat the distribution is.
  
  CASE 1: Very peaked distribution (high confidence context):
    P(token "Paris") = 0.9,  P(token "London") = 0.05, ...
    Top-k=50 → sampling from 50 tokens even though most probability is in 1-2 tokens
    → Many low-quality tokens (rank 3-50) can be sampled unnecessarily
  
  CASE 2: Flat distribution (high uncertainty context):
    P(token 1) = 0.05, P(token 2) = 0.049, ..., P(token 100) = 0.04
    Top-k=50 → cuts off many reasonable candidates at token 50
    → Arbitrary truncation, misses valid continuations

k-value sensitivity:
  k = 1:     greedy (deterministic)
  k = 10:    focused, low diversity
  k = 50:    moderate (standard)  ← GPT-2 default
  k = 200:   high diversity
  k = |V|:   no truncation (standard sampling)
```

## 3. Top-p (Nucleus) Sampling — Full Derivation

```
TOP-p (NUCLEUS) SAMPLING (Holtzman et al. 2019):
  Instead of a fixed count k, use a DYNAMIC vocabulary based on cumulative probability.

ALGORITHM:
  STEP 1: Sort tokens by probability (descending):
            π(1), π(2), ..., π(|V|) where P(π(1)) ≥ P(π(2)) ≥ ...
  
  STEP 2: Find nucleus size n*(p):
            n*(p) = min{n : ∑_{i=1}^{n} P(π(i)) ≥ p}
            (smallest set covering at least p of the probability mass)
  
  STEP 3: Sample from the nucleus:
            P_nucleus(j) = P(j) / ∑_{i ∈ nucleus} P(i)   for j in nucleus
            P_nucleus(j) = 0   otherwise

ADAPTIVE BEHAVIOR:
  High confidence context (P("Paris")=0.9, p=0.95):
    n*(0.95) = 2  (just "Paris" and one more token cover 95%)
    → Effectively k=2, very focused
  
  Low confidence context (P(j) ≈ 0.01 for many j, p=0.95):
    n*(0.95) = 95  (need 95 tokens to cover 95% probability)
    → Effectively k=95, wide vocabulary

MATHEMATICAL PROPERTY:
  The nucleus probability mass is EXACTLY ≥ p by construction.
  The normalised distribution over nucleus: ∑ P_nucleus = 1 ✓
  
  Expected quality of sampled token:
    E_{j~P_nucleus}[P_true(j)] = ∑_{j ∈ nucleus} P_nucleus(j) × P_true(j)
    
    For a good model (P_true ≈ P_model):
    E[P_true] ≈ ∑_{j ∈ nucleus} P(j)² / ∑_{i ∈ nucleus} P(i)  (higher than average P)
    → Nucleus sampling selects high-quality tokens while maintaining diversity

EFFECT OF p:
  p = 0.5: very small nucleus (greedy-like, focused)
  p = 0.9: standard choice (Holtzman et al. recommendation)
  p = 0.95: slightly more diverse
  p = 1.0: all tokens (equivalent to pure temperature sampling)
```

## 4. Min-p Sampling — New Alternative

```
MIN-p SAMPLING (2024, community invention):
  Dynamic cutoff based on a FRACTION of the maximum probability.

  Dynamic threshold:  θ(context) = p_base × P(most_likely_token)
  
  Include token j if P(j) ≥ θ(context)

ALGORITHM:
  p_max = max_j P(j)
  include j if P(j) ≥ p_base × p_max

NUMERICAL EXAMPLE:
  Logits give P = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]  (sorted desc)
  p_max = 0.5
  
  With p_base = 0.1:  threshold = 0.1 × 0.5 = 0.05
    Include tokens with P ≥ 0.05: {0.5, 0.3, 0.1, 0.05} ← 4 tokens
  
  With p_base = 0.1 in a flat distribution:
  P = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, ...] (flat)
  p_max = 0.1, threshold = 0.01
    Include tokens with P ≥ 0.01: many tokens ← wide distribution!

ADVANTAGE over Top-p:
  Top-p: nucleus grows linearly as distribution flattens.
  Min-p: nucleus grows in proportion to the RATIO p/p_max → more adaptive.
  
  Both adapt to distribution sharpness, but min-p uses relative rather than absolute threshold.
```

## 5. Combining Sampling Methods — Recommended Recipes

```
TYPICAL PRODUCTION CONFIGURATIONS:

For CREATIVE TASKS (story writing, dialogue):
  T = 0.7-1.0 + top_p = 0.9
  
  REASONING:
  T=0.9 adds noise → diverse generation
  top_p=0.9 removes the lowest-quality tokens → prevents garbage tokens
  
  Combined distribution:
  P_combined(j) ∝ exp(zⱼ/T) × 1[j ∈ nucleus(p=0.9)]

For FACTUAL TASKS (QA, code):
  T = 0.1-0.3 + top_k = 10-50
  
  REASONING:
  Low T → concentrated around best answers
  top_k = safety net to prevent degenerate low-probability samples

For INSTRUCTION FOLLOWING (chatbots):
  T = 0.6-0.8 + top_p = 0.9 (or similar)
  ← OpenAI API default: T=1.0 with top_p as the primary control

REPETITION PENALTY (frequency/presence penalty):
  Modify logits based on token frequency in already-generated text:
    z̃ⱼ = zⱼ − α × count(j in context)    (frequency penalty)
    z̃ⱼ = zⱼ − β × 1[j in context]        (presence penalty)
  
  MATHEMATICAL EFFECT:
    Each time token j is generated, its logit decreases by α (frequency) or β (presence).
    This makes the model AVOID repeating the same tokens.
    
  OPTIMAL α/β:
    α too small: repetition persists ("the the the" or paragraph loops)
    α too large: model avoids even necessary repetitions (can't refer to previous entities)
    
    Typical: α = 1.1-1.3, β = 0-0.2  for chat models
             α = 0 for code (variable names must repeat!)
```

---

## Exercises

1. Prove mathematically that as T → ∞, the softmax distribution approaches uniform. Show that limT→∞ exp(zk/T)/∑j exp(zj/T) = 1/|V|.
2. For P = [0.4, 0.3, 0.2, 0.06, 0.04] and p=0.9, find the nucleus manually. Compare to top-k=3. Which set is larger, and why does top-p adapt better here?
3. Compute the entropy H(P_T) for P=[0.6, 0.3, 0.1] at T=0.5, T=1.0, T=2.0. Show the monotonic relationship between T and H.
