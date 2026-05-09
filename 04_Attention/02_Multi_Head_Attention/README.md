# 02 — Multi-Head Attention (MHA)

## 1. Single vs Multi-Head — Expressiveness Proof

```
PROBLEM WITH SINGLE-HEAD ATTENTION:
  A single attention head computes ONE attention distribution over all positions.
  This gives ONE "view" of the context per position.
  
  For token "bank" in "I withdrew money from the bank":
    Single head: has to simultaneously represent:
      - Syntactic dependency (verb "withdrew" ← object)
      - Semantic context (financial institution)
    → One softmax over positions → can't attend to BOTH patterns simultaneously

MULTIPLE HEADS:
  h different attention functions in parallel, each with its own (Q,K,V) projections.
  Each head can learn a different attention pattern:
    Head 1: syntactic dependencies (next word)
    Head 2: coreference (pronoun → antecedent)
    Head 3: semantic similarity (related concepts)
    Head 4: positional (attend to recent tokens)
    ...

MATHEMATICAL EXPRESSIVENESS:
  Single head: can represent any single attention distribution (one softmax).
  Multi-head: can represent a MIXTURE of attention distributions.
  
  Output = ∑_h weight_h × value_h  (h parallel "lookups")
  
  The final linear projection W_O allows arbitrary combinations of head outputs.
  → Multi-head attention is strictly MORE expressive than single-head!
```

## 2. Full MHA Formulation

```
HYPERPARAMETERS: h heads, d_k = d_v = d/h (LLaMA-3 8B: h=32, d=4096, d_k=128)

PROJECTIONS per head i (i = 1, ..., h):
  Qᵢ = X · W_Q^i,   W_Q^i ∈ ℝ^{d × d_k}
  Kᵢ = X · W_K^i,   W_K^i ∈ ℝ^{d × d_k}
  Vᵢ = X · W_V^i,   W_V^i ∈ ℝ^{d × d_v}

HEAD COMPUTATION:
  headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ) = softmax(QᵢKᵢᵀ/√d_k + M) · Vᵢ

CONCATENATION and OUTPUT PROJECTION:
  MultiHead(X) = Concat(head₁, ..., headₕ) · W_O
                 ↑ shape: [n × (h·d_v)] = [n × d]   ← (since h·d_v = h·d/h = d)
  
  W_O ∈ ℝ^{d × d}  (the output projection mixes information across heads)

EQUIVALENT FORMULATION (all heads in parallel):
  Stack all head projections:
    W_Q ∈ ℝ^{d × d}  (block of all h Q-projection matrices)
    W_K ∈ ℝ^{d × d}
    W_V ∈ ℝ^{d × d}
  
  Compute Q, K, V ∈ ℝ^{n × d}  (all heads at once)
  Reshape to [n × h × d_k] (split into heads)
  Compute h separate attentions in parallel
  Reshape output to [n × d]
  Apply W_O
```

## 3. Parameter Count — Exact Calculation

```
MHA PARAMETERS (per layer):

  W_Q:  d × d  = 4096 × 4096 = 16,777,216
  W_K:  d × d  = 4096 × 4096 = 16,777,216    (MHA — separate for each head)
  W_V:  d × d  = 4096 × 4096 = 16,777,216
  W_O:  d × d  = 4096 × 4096 = 16,777,216
  ──────────────────────────────────────────
  Total:   67,108,864  ≈  67M per layer

  × 32 layers = 2.15B params in attention layers

  (LLaMA-3 8B uses GQA, not full MHA — see Ch.04.03 for GQA numbers)

COMPARISON:
  Full MHA (h=32, G=32):   67M per layer × 32 = 2.15B
  GQA (h=32, G=8):
    W_Q: 4096×4096 = 16.8M
    W_K: 4096×1024 = 4.2M  (G × d_k = 8 × 128 = 1024)
    W_V: 4096×1024 = 4.2M
    W_O: 4096×4096 = 16.8M
    Total per layer: 42M × 32 = 1.34B  (36% fewer params!)

COMPUTE PER LAYER:
  For sequence length n = 4096:
  
  QKV projections: 3 × 2 × n × d² = 6 × 4096 × 4096² = 412 GFLOPS
  Attention:       2 × n² × d_k × h = 2 × 4096² × 128 × 32 = 137 GFLOPS
  Output proj:     2 × n × d² = 137 GFLOPS
  ─────────────────────────────────────────────────────────────────
  Total attention: ~686 GFLOPS per layer
```

## 4. Attention Head Specialisation — Empirical Analysis

```
RESEARCH FINDING (Clark et al. 2019, Voita et al. 2019):
  Different heads specialise in different linguistic functions.
  
  HEAD TYPES identified:
  
  Type A: Positional Heads
    Attention pattern: strong diagonal offset
    Attends to [position i-1] or [position i+1]
    Implements: "look at the previous/next token"
    
    ┌ A matrix example ┐
    │ . 1 0 0 0        │  ← token 1 attends to token 2
    │ 0 . 1 0 0        │  ← token 2 attends to token 3
    │ 0 0 . 1 0        │
    └──────────────────┘

  Type B: Syntactic Heads
    Pattern: attends to syntactic head (verb from argument, modifier from head noun)
    Sparse pattern, long-range dependencies
    
    Example: "The cat that the dog bit ran away"
    "ran" attends strongly to "cat" (subject), not to "bit" or "dog"

  Type C: Rare Token/BOS Heads (Jain & Wallace 2019)
    Many heads dump large attention weight on BOS (first token)
    This is a "no-op" — the BOS token acts as a garbage collector
    Occurs when the head has nothing relevant to attend to
    
    Mathematical: if all position scores are low, softmax distributes equally.
    BOS gets extra attention because it often appears in training without masking.
    
  Type D: Global Context Heads
    Relatively uniform attention distribution
    H_entropy ≈ log(n) (maximum entropy)
    Compute soft average of all context tokens

MATHEMATICAL EVIDENCE FOR SPECIALISATION:
  Inter-head diversity: different heads → different attention matrices.
  
  For heads i and j:
    Divergence(Aᵢ, Aⱼ) = KL(row distributions of Aᵢ ‖ row distributions of Aⱼ)
    High divergence → different patterns → good diversity (each head unique)
    Low divergence → similar patterns → redundant heads
  
  Pruning experiments show: ~30-50% of heads can be removed with <1% accuracy drop!
  (Michel et al. 2019: "Are Sixteen Heads Really Better Than One?")
```

## 5. Efficient Implementation — Batched Computation

```
NAÏVE IMPLEMENTATION:
  Loop over h heads, compute attention separately for each.
  → Sequential, misses GPU parallelism.

EFFICIENT: Batch across heads.
  
  Reshape Q, K, V to [batch, heads, seq_len, d_k]:
    Q: [B, n, d] → [B, n, h, d_k] → [B, h, n, d_k]
    K: [B, n, d] → [B, n, h, d_k] → [B, h, n, d_k]
    V: [B, n, d] → [B, n, h, d_v] → [B, h, n, d_v]
  
  Batched matmul: Scores = Q @ Kᵀ  (PyTorch: torch.bmm treats first 2 dims as batch)
    Shape: [B, h, n, d_k] @ [B, h, d_k, n] = [B, h, n, n]
  
  Softmax: applied along last dimension (over n keys)
  
  Output: Scores @ V → [B, h, n, d_v]
  
  Reshape back: [B, h, n, d_v] → [B, n, h×d_v] = [B, n, d]
  
  Apply W_O: [B, n, d] @ [d, d] = [B, n, d]

PYTORCH ONE-LINER (modern versions):
  output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)
  → Automatically uses FlashAttention if available → 2-4× faster + memory efficient!
```

---

## Exercises

1. Verify that h=32 heads with d_k=128 gives d_k × h = d = 4096 (the standard dimension reduction in MHA).
2. Show that single-head attention (h=1, d_k=d) is a special case of multi-head attention. What is W_O in this case?
3. For d=512, h=8, n=128: compute the number of FLOPs for the attention matrix computation (Q·Kᵀ) and the weighted sum (A·V). Which dominates for n >> d_k?
