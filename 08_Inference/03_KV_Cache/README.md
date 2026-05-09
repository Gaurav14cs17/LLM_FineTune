# Understanding the KV Cache

*From the autoregressive redundancy problem to full KV cache implementation and memory formulas*

---

**The KV Cache** is the single most important inference optimisation for LLMs. Without it, generating 100 tokens takes 100× more compute than the first token. With it, each new token requires roughly constant compute regardless of context length.

This guide explains why the cache is necessary, exactly what it stores, and how to compute its memory footprint.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 The Redundancy Problem](#11-the-redundancy-problem)
   - [1.2 What the KV Cache Stores](#12-what-the-kv-cache-stores)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The Math of Autoregressive Inference](#2-the-math-of-autoregressive-inference)
   - [2.1 Without KV Cache: O(n²) cost](#21-without-kv-cache-on-cost)
   - [2.2 With KV Cache: O(n) cost](#22-with-kv-cache-on-cost)
3. [Memory Footprint](#3-memory-footprint)
   - [3.1 KV Cache Size Formula](#31-kv-cache-size-formula)
   - [3.2 LLaMA-3 8B Numbers](#32-llama-3-8b-numbers)
4. [Optimised KV Attention: GQA](#4-optimised-kv-attention-gqa)
   - [4.1 MHA vs GQA vs MQA](#41-mha-vs-gqa-vs-mqa)
   - [4.2 Memory and Speed Comparison](#42-memory-and-speed-comparison)
5. [Advanced: PagedAttention](#5-advanced-pagedattention)
6. [Summary](#6-summary)
   - [6.1 Formulas Quick Reference](#61-formulas-quick-reference)
   - [6.2 Common Mistakes](#62-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 The Redundancy Problem

```
┌─────────────────────────────────────────────────────────────┐
│  AUTOREGRESSIVE GENERATION: tokens generated one at a time  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Input ["The"] → generate "cat"                     │
│    Compute: Q["The"], K["The"], V["The"]  ← 1 token         │
│                                                             │
│  Step 2: Input ["The", "cat"] → generate "sat"              │
│    Compute: Q["The"], K["The"], V["The"]  ← RECOMPUTED!     │
│             Q["cat"], K["cat"], V["cat"]  ← new token       │
│                                                             │
│  Step 3: Input ["The","cat","sat"] → generate "on"          │
│    Compute: Q["The"], K["The"], V["The"]  ← RECOMPUTED!     │
│             Q["cat"], K["cat"], V["cat"]  ← RECOMPUTED!     │
│             Q["sat"], K["sat"], V["sat"]  ← new token       │
│                                                             │
│  At step n: recompute ALL previous n-1 tokens' K, V!       │
│  Total FLOPs: 1 + 2 + 3 + ... + n = O(n²)                  │
│                                                             │
│  For n=1024: 524,288 total operations instead of 1,024!    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 What the KV Cache Stores

```
┌─────────────────────────────────────────────────────────────┐
│  KEY INSIGHT:                                               │
│  K and V vectors for past tokens NEVER CHANGE.             │
│  Once computed, they are identical at every future step.   │
│                                                             │
│  Q changes at every step (new query = new current token)   │
│  K[t] is fixed forever once position t is processed         │
│  V[t] is fixed forever once position t is processed         │
│                                                             │
│  SOLUTION: Save K and V into a cache. At each new step:    │
│  1. Only compute Q, K, V for the NEW token (position n)    │
│  2. Append K[n] and V[n] to the cache                       │
│  3. Compute attention: Q[n] × [K_cache ; K[n]]ᵀ           │
│  4. Weighted sum over [V_cache ; V[n]]                      │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Reading a book: without the cache, you re-read every previous page before reading the next one. With the cache, you keep a bookmark (saved notes) and only read the new page.

### 1.3 Pipeline Summary

```
PREFILL PHASE (processing the prompt):
  Input: [w₁, w₂, ..., w_P]  (P prompt tokens, processed in PARALLEL)
  Compute: K[1..P] and V[1..P] for ALL layers
  Store: cache[l][K] = K_l[1..P],  cache[l][V] = V_l[1..P]
  Cost: O(P²) per layer  (full attention over prompt)

DECODE PHASE (generating new tokens):
  For each step t = P+1, P+2, ...:
    Input: [w_t]  (ONE new token)
    Compute: K_l[t], V_l[t] for ALL layers L
    Append: cache[l][K] += K_l[t],  cache[l][V] += V_l[t]
    Attention: Q_l[t] × cache[l][K]ᵀ / √d_k  (Q is [1 × d_k])
    Cost: O(t) per layer  (one query attends to all t past keys)
  ↓
TOTAL DECODE COST for generating G tokens:
  O(L × (P + 1) + P + 2) + ... + (P + G))
= O(L × G × P)   → LINEAR in context length
```

---

## 2. The Math of Autoregressive Inference

### 2.1 Without KV Cache: O(n²) cost

```
At decode step t (total context length = t):
  ALL tokens need full re-computation:

  For each token position s ∈ {1,...,t}:
    x_s = embed(w_s)
    For each layer l ∈ {1,...,L}:
      K_l[s] = x_l[s] × W_K    ← RECOMPUTED at every step!
      V_l[s] = x_l[s] × W_V    ← RECOMPUTED at every step!
      Q_l[s] = x_l[s] × W_Q    ← RECOMPUTED (but not needed for past s)

  Total FLOPs for decode step t:
    F(t) ≈ 2 × L × t × d × (3d_k + d_v)   (all t tokens)

  Total FLOPs for all G decode steps (t = P+1 to P+G):
    Σₜ F(t) ≈ 2L × Σₜ₌ₚ₊₁^{P+G} t × (...) ≈ O(L × G² × d²/G)
            ≈ O(L × G × d²)  but with GROWING cost per step!

  ┌─────────────────────────────────────────────────────────────┐
  │  GROWING COST per decode step (L=32, d=4096, d_k=128):     │
  │  t=512:   ~2×10¹¹ FLOPs (0.2 TFLOPs)                      │
  │  t=1024:  ~4×10¹¹ FLOPs (0.4 TFLOPs)  ← 2× slower!        │
  │  t=4096:  ~1.6×10¹² FLOPs (1.6 TFLOPs) ← 8× slower!       │
  └─────────────────────────────────────────────────────────────┘
```

### 2.2 With KV Cache: O(n) cost

```
At decode step t:
  Only the NEW token w_t is processed:

  For new token w_t:
    x = embed(w_t)
    For each layer l:
      K_new = x_l × W_K     ← ONE token
      V_new = x_l × W_V     ← ONE token
      Q_new = x_l × W_Q     ← ONE token
      
      Attention: Q_new × [K_cache ; K_new]ᵀ / √d_k
                         ↑ t-1 cached keys + 1 new key = t total
      
      FLOPs: 2×d×d_k (Q,K,V proj) + 2×t×d_k (attention dot products)

  COST per decode step t:
    F_kv(t) ≈ 2 × L × d × (3d_k + t)    ← t appears, but just once!

  COMPARISON:
  ┌─────────────────────────────────────────────────────────────┐
  │  t=512:  No KV:  2×10¹¹ FLOPs   KV:  ~3×10⁹ FLOPs  (67×)  │
  │  t=1024: No KV:  4×10¹¹ FLOPs   KV:  ~6×10⁹ FLOPs  (67×)  │
  │  t=4096: No KV:  1.6×10¹² FLOPs KV:  ~2.4×10¹⁰ FLOPs (67×)│
  │                                                             │
  │  The KV cache speedup is roughly constant (60-100×)!        │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Memory Footprint

### 3.1 KV Cache Size Formula

```
For each token in context:
  We store: K_l ∈ ℝ^{h × d_k}  and  V_l ∈ ℝ^{h × d_v}
  Per token per layer: 2 × h × d_k values (assuming d_k=d_v)

Total KV cache size:
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  M_KV = 2 × n × L × h × d_k × bytes_per_value           │
  │                                                          │
  │  2:   K and V                                            │
  │  n:   number of tokens in context                        │
  │  L:   number of layers                                   │
  │  h:   number of attention heads (for K and V)            │
  │  d_k: dimension per head                                 │
  │  bytes_per_value: 2 for BF16/FP16, 1 for INT8            │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

### 3.2 LLaMA-3 8B Numbers

```
LLaMA-3 8B configuration:
  L = 32 layers
  h = 8 KV heads  (GQA: 32 query heads, 8 KV heads)
  d_k = 128 (head dimension)
  dtype = BF16 (2 bytes)

KV cache per token:
  = 2 × L × h_kv × d_k × 2 bytes
  = 2 × 32 × 8 × 128 × 2
  = 131,072 bytes = 128 KB per token

MEMORY FOR DIFFERENT CONTEXT LENGTHS:
  ┌─────────────────────────────────────────────────────────┐
  │  Context length n    KV cache size                      │
  │  ────────────────────────────────────────               │
  │  1,024 tokens        128 KB × 1K  = 0.13 GB            │
  │  4,096 tokens        128 KB × 4K  = 0.52 GB            │
  │  8,192 tokens        128 KB × 8K  = 1.05 GB   ← std    │
  │  32,768 tokens       128 KB × 32K = 4.19 GB            │
  │  131,072 tokens      128 KB ×128K = 16.8 GB   ← GQA!  │
  └─────────────────────────────────────────────────────────┘

WITHOUT GQA (32 KV heads instead of 8):
  128K context = 16.8 GB × 4 = 67 GB  → exceeds GPU memory!
  GQA SAVES 75% of KV cache memory while losing minimal quality.
```

---

## 4. Optimised KV Attention: GQA

### 4.1 MHA vs GQA vs MQA

```
┌─────────────────────────────────────────────────────────────┐
│  MULTI-HEAD ATTENTION (MHA):                                │
│    Each query head has its own K, V heads                   │
│    h_q = h_k = h_v = H   (all equal, e.g., H=32)           │
│                                                             │
│  Q  K  V pairs: 32 × 32 × 32                               │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬── ... ──┬──┬──┬──┐             │
│  │Q₁│K₁│V₁│Q₂│K₂│V₂│Q₃│K₃│V₃ ...  │Q₃₂│K₃₂│V₃₂│           │
│  └──┴──┴──┴──┴──┴──┴──┴──┴── ... ──┴──┴──┴──┘             │
│                                                             │
│  GROUPED QUERY ATTENTION (GQA):  Used in LLaMA-3, Mistral  │
│    G query heads share ONE K, V head (G = h_q / h_kv)      │
│    h_q = 32, h_kv = 8 → G = 4 queries share 1 KV pair      │
│                                                             │
│  ┌──┬──┬──┬──┬─────┬──┬──┬──┬──┬─────┬── ... ─────┐       │
│  │Q₁│Q₂│Q₃│Q₄│K₁V₁│Q₅│Q₆│Q₇│Q₈│K₂V₂│  ...  K₈V₈│       │
│  └──┴──┴──┴──┴─────┴──┴──┴──┴──┴─────┴── ... ─────┘       │
│                                                             │
│  MULTI-QUERY ATTENTION (MQA):  Used in early PaLM           │
│    ALL query heads share a SINGLE K, V head                 │
│    h_q = 32, h_kv = 1                                       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Memory and Speed Comparison

```
For LLaMA-3 8B at context=8192, BF16:

  MHA (32 KV heads):  2 × 8192 × 32 × 32 × 128 × 2 = 4.29 GB
  GQA (8 KV heads):   2 × 8192 × 32 × 8  × 128 × 2 = 1.07 GB  ← 4× less!
  MQA (1 KV head):    2 × 8192 × 32 × 1  × 128 × 2 = 0.13 GB  ← 32× less!

QUALITY vs MEMORY TRADE-OFF:
  MHA:  Best quality (each head has its own keys to attend to)
  GQA:  95-98% of MHA quality, 4× less memory  ← sweet spot
  MQA:  90-95% of MHA quality, 32× less memory (some degradation)

WHY GQA DOESN'T HURT MUCH:
  Empirically, attention heads within a group tend to learn similar patterns.
  Sharing K, V across 4 heads loses very little representational power
  while dramatically reducing the memory bandwidth bottleneck at inference.
```

---

## 5. Advanced: PagedAttention

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM WITH CONTIGUOUS KV CACHE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRADITIONAL APPROACH:                                      │
│  Pre-allocate max_seq_len × kv_size for each request        │
│                                                             │
│  If max_seq_len = 4096, but request only uses 100 tokens:   │
│  → 4096/100 - 1 = 39× memory WASTED (internal fragmentation)│
│                                                             │
│  GPU memory: 80 GB total                                    │
│  With 10 concurrent requests: each gets 8 GB max            │
│  → Can serve requests up to 8 GB / (128 KB/tok) = 65K tokens│
│  But most requests are short → 90% of that 8 GB is wasted! │
└─────────────────────────────────────────────────────────────┘

PAGEDATTENTION (vLLM, 2023):

  INSPIRED BY: OS virtual memory paging

  Instead of contiguous blocks:
  ┌─────────────────────────────────────────────────────────┐
  │  KV cache split into PAGES of fixed size B tokens each  │
  │                                                         │
  │  Request 1 (100 tokens):  pages [3, 7, 1]               │
  │  Request 2 (200 tokens):  pages [0, 4, 8, 2, 5, 9]     │
  │  Request 3 (50  tokens):  pages [6]                     │
  │                                                         │
  │  Physical pages allocated ON-DEMAND as sequence grows   │
  │  Pages can be non-contiguous in GPU memory              │
  │                                                         │
  │  RESULT: <5% waste vs 30-50% in standard approach!      │
  │  → 2-4× more requests in memory simultaneously          │
  │  → 2-4× higher throughput                               │
  └─────────────────────────────────────────────────────────┘

  ATTENTION WITH PAGED MEMORY:
    Block table maps (sequence position) → (physical page, offset)
    Gather K, V from non-contiguous pages before attention
    Overhead: ~5-10% slower per operation, but 2-4× better throughput
```

---

## 6. Summary

### 6.1 Formulas Quick Reference

**KV Cache Memory:**

```
M_KV = 2 × n × L × h_kv × d_k × bytes_per_value

For LLaMA-3 8B BF16 (h_kv=8, d_k=128, L=32):
  M_KV(n) = 128 KB × n  (per token)
```

**FLOPs without KV cache (step t):**

```
F_no_cache(t) = O(L × t × d²)   (recompute all t tokens)
```

**FLOPs with KV cache (step t):**

```
F_kv(t) = O(L × (d² + t × d_k))   (new token + attend to cache)
```

**GQA groups:**

```
G = h_q / h_kv  (query heads per KV head)
LLaMA-3 8B: G = 32/8 = 4
```

| Method | KV Cache Size | Quality | Notes |
|--------|--------------|---------|-------|
| MHA | H × base | Best | Original Transformer |
| GQA (G=4) | H/4 × base | ~97% | LLaMA-3, Mistral |
| MQA | H/H × base | ~92% | PaLM, Falcon-7B |
| GQA (G=2) | H/2 × base | ~99% | Balance |

### 6.2 Common Mistakes

```
❌ WRONG: The KV cache stores Q vectors too
✓ RIGHT:  Only K and V are cached. Q is computed fresh at each step
          because Q changes with each new input token. K and V for
          past tokens are constant (they don't depend on future tokens).

❌ WRONG: KV cache can be used during training
✓ RIGHT:  During training, ALL positions are processed in parallel and
          all positions need gradients through the full attention matrix.
          KV cache is an INFERENCE-ONLY optimization.

❌ WRONG: GQA just reduces quality for memory savings
✓ RIGHT:  Empirical results show GQA (G=4-8) loses < 1% quality vs MHA
          while reducing KV cache by 4-8×. The trade-off is highly favourable.

❌ WRONG: PagedAttention changes the attention computation
✓ RIGHT:  PagedAttention changes MEMORY MANAGEMENT, not the math.
          The attention output is identical to standard attention.
          Only the physical layout of K, V in GPU memory differs.
```

---

## 7. Exercises

1. **Memory Calculation**: For LLaMA-3 70B (L=80, h_kv=8, d_k=128) with BF16, compute the KV cache size for n=8192 tokens. How many 80 GB A100s would be needed for this context window alone?

2. **FLOPs Comparison**: For a 32-layer model with d=4096, d_k=128, compare FLOPs for generating the 500th token (a) without KV cache (full re-computation), (b) with KV cache. What is the speedup ratio?

3. **GQA Memory**: A model has 32 query heads and d_k=128. Compute KV cache per token (in KB, BF16) for MHA (h_kv=32), GQA with G=4 (h_kv=8), and MQA (h_kv=1). For n=32K context, what is the total cache in GB for each?

4. **PagedAttention**: If page size B=16 tokens and a request has 1537 tokens: how many pages are allocated? What is the internal fragmentation (wasted space) in tokens? Compare to allocating max_seq_len=2048 upfront.
