# Understanding Scaled Dot-Product Attention

*A comprehensive guide from intuition to full mathematical proof*

---

**Scaled Dot-Product Attention** is the core computational primitive of every Transformer-based LLM. It allows each token to "look at" all other tokens in the context and gather relevant information.

This guide walks from the soft-lookup intuition through the complete mathematical derivation, variance proofs, and the FlashAttention memory optimisation.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What is Attention?](#11-what-is-attention)
   - [1.2 Soft vs Hard Lookup](#12-soft-vs-hard-lookup)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The Formula](#2-the-formula)
   - [2.1 Full Mathematical Formulation](#21-full-mathematical-formulation)
   - [2.2 Step-by-Step Computation](#22-step-by-step-computation)
3. [The √d_k Scaling](#3-the-dk-scaling)
   - [3.1 Variance Proof](#31-variance-proof)
   - [3.2 Effect on Softmax](#32-effect-on-softmax)
4. [Causal Masking](#4-causal-masking)
   - [4.1 Why Causal Masking?](#41-why-causal-masking)
   - [4.2 Proof of Causality](#42-proof-of-causality)
5. [Attention Entropy](#5-attention-entropy)
6. [FlashAttention](#6-flashattention)
   - [6.1 The Memory Problem](#61-the-memory-problem)
   - [6.2 Online Softmax Algorithm](#62-online-softmax-algorithm)
7. [Summary](#7-summary)
   - [7.1 All Formulas Quick Reference](#71-all-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 What is Attention?

Attention is a **differentiable, soft dictionary lookup**. Given a query, it searches all key-value pairs and returns a weighted blend of values.

### 1.2 Soft vs Hard Lookup

```
┌─────────────────────────────────────────────────────────────┐
│  HARD dictionary lookup (hash map):                         │
├─────────────────────────────────────────────────────────────┤
│  Query: "cat"   →  find exact key "cat"  →  return value    │
│                                                             │
│  ┌──────┬─────────────────┐                                 │
│  │ Key  │ Value           │                                 │
│  ├──────┼─────────────────┤                                 │
│  │ cat  │ "furry animal"  │  ← exact match ONLY             │
│  │ dog  │ "canine"        │  ← ignored                      │
│  │ fish │ "swims"         │  ← ignored                      │
│  └──────┴─────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SOFT attention lookup (Transformer):                       │
├─────────────────────────────────────────────────────────────┤
│  Query: "feline"  →  score ALL keys  →  weighted blend      │
│                                                             │
│  ┌──────┬───────────────┬──────────────────┐               │
│  │ Key  │ Score (q·k)   │ Weight (softmax) │               │
│  ├──────┼───────────────┼──────────────────┤               │
│  │ cat  │  qᵀk_cat=2.1  │      0.65        │               │
│  │ dog  │  qᵀk_dog=0.5  │      0.20        │               │
│  │ fish │  qᵀk_fish=-1  │      0.08        │               │
│  │ lion │  qᵀk_lion=0.8 │      0.07        │               │
│  └──────┴───────────────┴──────────────────┘               │
│                                                             │
│  Output = 0.65×V_cat + 0.20×V_dog + 0.08×V_fish + ...     │
│         = blended representation weighted by relevance     │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Imagine searching for "feline" in a library. A hard lookup fails (no book titled "feline"). A soft lookup returns the "cat" book mostly, with some "lion" and "tiger" mixed in — weighted by similarity.

### 1.3 Pipeline Summary

```
INPUT: token sequence → hidden states X ∈ ℝⁿˣᵈ
        ↓
Step 1: Linear projections Q = XW_Q,  K = XW_K,  V = XW_V
        ↓
Step 2: Score matrix  S = QKᵀ / √d_k       ← raw similarities
        ↓
Step 3: Mask (add −∞ to future positions)  S̃ = S + M
        ↓
Step 4: Softmax  A = softmax(S̃)            ← attention weights
        ↓
Step 5: Weighted sum  Z = A × V             ← output
        ↓
OUTPUT: Z ∈ ℝⁿˣᵈᵥ  (contextualised token representations)
```

---

## 2. The Formula

### 2.1 Full Mathematical Formulation

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│              ⎛   Q Kᵀ      ⎞                                     │
│  Attention = softmax ⎜ ─────── + M ⎟  ×  V                      │
│              ⎝   √d_k      ⎠                                     │
│                                                                  │
│  where:                                                          │
│    Q ∈ ℝⁿˣᵈᵏ  (query matrix)                                    │
│    K ∈ ℝⁿˣᵈᵏ  (key matrix)                                      │
│    V ∈ ℝⁿˣᵈᵛ  (value matrix)                                    │
│    M ∈ {0, −∞}ⁿˣⁿ  (causal mask)                               │
│    d_k  (key dimension, used for scaling)                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Variables:**

| Symbol | Meaning | LLaMA-3 8B value |
|--------|---------|------------------|
| n | Sequence length | up to 8192 |
| d | Model hidden size | 4096 |
| d_k | Key/query dimension per head | 128 |
| d_v | Value dimension per head | 128 |
| h | Number of attention heads | 32 |
| M | Causal mask (−∞ for future) | lower-triangular |

### 2.2 Step-by-Step Computation

**Step 1 — Linear Projections:**

```
For input X ∈ ℝⁿˣᵈ (n tokens, each d-dimensional):

  Q = X × W_Q       W_Q ∈ ℝᵈˣᵈᵏ   → Q ∈ ℝⁿˣᵈᵏ
  K = X × W_K       W_K ∈ ℝᵈˣᵈᵏ   → K ∈ ℝⁿˣᵈᵏ
  V = X × W_V       W_V ∈ ℝᵈˣᵈᵛ   → V ∈ ℝⁿˣᵈᵛ
```

**Step 2 — Score Matrix:**

```
S = Q × Kᵀ / √d_k         S ∈ ℝⁿˣⁿ

S[i, j] = qᵢ · kⱼ / √d_k
         = (similarity between query at position i and key at position j)
```

**Step 3 — Causal Mask:**

```
S̃[i, j] = S[i, j]   if j ≤ i  (position j is in the past or present)
S̃[i, j] = −∞        if j > i  (position j is in the future — forbidden)
```

**Step 4 — Softmax (row-wise):**

```
A[i, j] = exp(S̃[i,j]) / Σₖ exp(S̃[i,k])

Properties:
  A[i, j] ≥ 0            (non-negative)
  Σⱼ A[i, j] = 1         (each row sums to 1)
  A[i, j] = 0 if j > i   (causal: no future attention)
```

**Step 5 — Weighted Sum:**

```
Z[i] = Σⱼ A[i, j] × V[j]
     = A[i, :] × V

"The output at position i is a weighted blend of all past value vectors,
weighted by how relevant each past token is to the current query."
```

**Concrete Numerical Example (n=3, d_k=2):**

```
Q = [[1, 0],    K = [[1, 0],    V = [[10, 0],
     [0, 1],         [0, 1],         [0, 10],
     [1, 1]]         [1, 1]]         [5,  5]]

Step 2: S = Q × Kᵀ / √2

S[0,0] = (1×1 + 0×0) / 1.41 = 0.71
S[0,1] = (1×0 + 0×1) / 1.41 = 0.00
S[0,2] = (1×1 + 0×1) / 1.41 = 0.71
...

Step 3: Apply causal mask (n=3):
  Row 0 can only see position 0: mask S[0,1] = −∞, S[0,2] = −∞
  Row 1 can see positions 0,1:   mask S[1,2] = −∞
  Row 2 can see all positions:   no masking

Step 4: Softmax (row 0): A[0] = softmax([0.71, −∞, −∞]) = [1.0, 0, 0]

Step 5: Z[0] = 1.0 × V[0] = [10, 0]   (token 0 only sees itself)
```

---

## 3. The √d_k Scaling

### 3.1 Variance Proof

```
┌─────────────────────────────────────────────────────────────┐
│  CLAIM: Without scaling, Var[q·k] = d_k                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROOF:                                                     │
│  Assume qᵢ ~ N(0,1) and kᵢ ~ N(0,1) independently          │
│                                                             │
│  s = q·k = Σᵢ₌₁^{d_k} qᵢ × kᵢ                             │
│                                                             │
│  E[s] = Σᵢ E[qᵢkᵢ] = Σᵢ E[qᵢ]E[kᵢ] = Σᵢ 0×0 = 0          │
│                                                             │
│  Var[s] = Var[Σᵢ qᵢkᵢ]                                     │
│         = Σᵢ Var[qᵢkᵢ]          (independence)             │
│         = Σᵢ (E[qᵢ²k²ᵢ] − 0)                               │
│         = Σᵢ E[qᵢ²]E[k²ᵢ]       (independence)             │
│         = Σᵢ 1 × 1                                         │
│         = d_k   ■                                           │
│                                                             │
│  After dividing by √d_k:                                    │
│  Var[s / √d_k] = Var[s] / d_k = d_k / d_k = 1   ✓         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Effect on Softmax

```
BEFORE SCALING (d_k = 64):
  std[s] = √64 = 8
  Typical scores: [-8, -4, 0, 4, 8, 12, ...]

  softmax([0, 4, 8, 12]):
    exp values: [1.0, 54.6, 2981, 162755]
    weights: [0.0001, 0.003, 0.018, 0.979]   ← nearly one-hot!

  ┌─────────────────────────────────────────────────────────┐
  │  token 1  │                                 0.000       │
  │  token 2  │                                 0.003       │
  │  token 3  │                                 0.018       │
  │  token 4  │█████████████████████████████████ 0.979      │
  └─────────────────────────────────────────────────────────┘
  → Gradient of softmax ≈ 0 → VANISHING GRADIENTS!

AFTER SCALING by 1/√64 = 1/8:
  Typical scores: [-1, -0.5, 0, 0.5, 1, 1.5, ...]
  
  softmax([0, 0.5, 1.0, 1.5]):
    exp values: [1.0, 1.65, 2.72, 4.48]
    weights: [0.10, 0.17, 0.28, 0.45]   ← smooth, distributed!

  ┌─────────────────────────────────────────────────────────┐
  │  token 1  │██████████                        0.10       │
  │  token 2  │█████████████████                 0.17       │
  │  token 3  │████████████████████████████      0.28       │
  │  token 4  │████████████████████████████████  0.45       │
  └─────────────────────────────────────────────────────────┘
  → Smooth gradients → stable training ✓
```

---

## 4. Causal Masking

### 4.1 Why Causal Masking?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION: No Peeking Ahead!                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  During TRAINING on "The cat sat":                          │
│  Position 1 ("cat") should NOT see position 2 ("sat")       │
│  Why? We're training the model to PREDICT "sat" from "cat"  │
│  If "cat" can see "sat", it cheats — no learning happens!   │
│                                                             │
│  The causal mask enforces:                                  │
│  "Position i can only attend to positions 1, 2, ..., i"    │
│                                                             │
│  MASK MATRIX for n=4:                                       │
│       k=1   k=2   k=3   k=4                                │
│  q=1  [  0    -∞   -∞    -∞ ]  ← position 1: sees only 1  │
│  q=2  [  0     0   -∞    -∞ ]  ← position 2: sees 1,2     │
│  q=3  [  0     0    0    -∞ ]  ← position 3: sees 1,2,3   │
│  q=4  [  0     0    0     0 ]  ← position 4: sees all     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Proof of Causality

**Theorem:** Output Z[i] depends only on input tokens x[1], ..., x[i].

**Proof:**

```
Step 1: After masking, softmax gives A[i, j] = 0 for j > i.
        (because exp(−∞) = 0, so those weights are exactly 0)

Step 2: Z[i] = Σⱼ A[i,j] × V[j]
             = Σⱼ≤ᵢ A[i,j] × V[j]  +  Σⱼ>ᵢ 0 × V[j]
             = Σⱼ≤ᵢ A[i,j] × V[j]

Step 3: V[j] = x[j] × W_V → depends only on x[j]

Step 4: Therefore Z[i] depends only on x[1], ..., x[i].   ■

Key benefit: DURING TRAINING, all positions 1..n are computed in ONE
parallel matrix multiplication — no sequential loop needed!
During inference, we generate one token at a time using the KV cache.
```

---

## 5. Attention Entropy

Attention entropy measures how "focused" or "diffuse" an attention pattern is:

```
H_i = −Σⱼ A[i,j] × log A[i,j]

Range: 0 ≤ H_i ≤ log(i)   (max entropy for i available positions)

INTERPRETATION:
  H_i = 0:          completely focused attention (A[i,j]=1 for one j)
  H_i = log(i):     uniform attention (A[i,j] = 1/i for all j ≤ i)

OBSERVED HEAD PATTERNS (empirically):

  Head Type A — Positional (low entropy):
  ┌──────────────────────────────────────────────────────────┐
  │  Position: 1  2  3  4  5  6                              │
  │  Weight:  .00 .98 .01 .00 .00 .00  ← looks at prev token│
  │  H ≈ 0.1  (very focused)                                 │
  └──────────────────────────────────────────────────────────┘

  Head Type B — Global context (high entropy):
  ┌──────────────────────────────────────────────────────────┐
  │  Position: 1  2  3  4  5  6                              │
  │  Weight:  .17 .17 .17 .16 .17 .16  ← uniform blend      │
  │  H ≈ log(6) = 1.79  (maximum entropy)                   │
  └──────────────────────────────────────────────────────────┘
```

---

## 6. FlashAttention

### 6.1 The Memory Problem

```
┌─────────────────────────────────────────────────────────────┐
│  NAIVE ATTENTION MEMORY COST                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Score matrix S ∈ ℝⁿˣⁿ                                     │
│    n=4096, float32 → 4096² × 4 bytes = 67 MB per head!     │
│    With h=32 heads × L=32 layers: 67 MB × 1024 = 68 GB    │
│    → Exceeds A100 memory → CANNOT train on long sequences  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

KEY INSIGHT: We never actually NEED the full n×n matrix.
We only need the final output Z = softmax(QKᵀ/√d) × V.
Can we compute Z without materialising QKᵀ? YES — with tiling!
```

### 6.2 Online Softmax Algorithm

FlashAttention uses an **online (incremental) softmax** that processes blocks:

```
STANDARD softmax requires 2 passes:
  Pass 1: find max m = max(aᵢ)    (for numerical stability)
  Pass 2: compute exp(aᵢ−m) / Σ exp(aⱼ−m)

ONLINE softmax (one pass, processes blocks):
  After seeing values [a₁,...,aₗ]:
    mₗ = max(a₁,...,aₗ)                    running max
    dₗ = Σⱼ₌₁ˡ exp(aⱼ − mₗ)              running denominator
  
  When new value aₗ₊₁ arrives:
    mₗ₊₁ = max(mₗ, aₗ₊₁)
    dₗ₊₁ = dₗ × exp(mₗ − mₗ₊₁) + exp(aₗ₊₁ − mₗ₊₁)
                  ↑                         ↑
            rescale old terms         add new term
  
  Final: softmax_i = exp(aᵢ − m_n) / d_n

MEMORY REDUCTION:
  FlashAttention:  O(n) memory   (only blocks fit in SRAM)
  Naive attention: O(n²) memory  (full matrix in HBM)
  
  For n=4096, d=128:
    Naive:  67 MB per head
    Flash:  ~1 MB per head   (67× reduction!)

SPEED:
  A100 SRAM bandwidth: 19 TB/s  (10× faster than HBM)
  FlashAttention keeps computation in SRAM → 2-4× wall-clock speedup
```

---

## 7. Summary

### 7.1 All Formulas Quick Reference

**Projections:**

```
Q = X × W_Q,   K = X × W_K,   V = X × W_V
```

**Attention:**

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k  +  M)  ×  V
```

**Score variance proof:**

```
Var[qᵢ·kᵢ] = d_k   (without scaling)
Var[qᵢ·kᵢ / √d_k] = 1   (with scaling)  ← unit variance
```

**Causal mask:**

```
M[i,j] = 0     if j ≤ i
M[i,j] = −∞   if j > i
```

**Attention entropy:**

```
H_i = −Σⱼ A[i,j] log A[i,j],   0 ≤ H_i ≤ log(i)
```

| Component | Purpose | Without it |
|-----------|---------|-----------|
| Q K projections | Learn what to query / what to offer | Can't adapt to content |
| 1/√d_k scaling | Keep variance = 1 after dot product | Vanishing gradients |
| Causal mask | No future peeking during training | Trivial solution (copy answer) |
| Softmax | Produce valid probability weights | Negative weights possible |
| V projection | Learn what information to retrieve | Limited expressivity |

### 7.2 Common Mistakes

```
❌ WRONG: Dividing by d_k instead of √d_k
✓ RIGHT:  The scaling factor is 1/√d_k (square root!)
          Reason: we want Var = 1, and Var[s]=d_k, so divide by √d_k.

❌ WRONG: Applying softmax over the full matrix (including masked positions)
✓ RIGHT:  exp(−∞) = 0, so masked positions get exactly 0 weight after softmax.
          This is why adding −∞ (not a large finite number) is critical.

❌ WRONG: Sharing Q, K, V projections (i.e. W_Q = W_K)
✓ RIGHT:  They are separate learned matrices. Q and K can differ significantly.
          (GQA reuses K and V across heads, but Q is always separate.)

❌ WRONG: Attention output shape = [n × d_k]
✓ RIGHT:  Attention output shape = [n × d_v], which equals [n × d_k] when d_v = d_k
          but they don't have to be equal.
```

---

## 8. Exercises

1. **Variance Proof**: For d_k = 256, compute Var[q·k] without scaling. What would the typical range of scores be (±3σ)? Why does this cause softmax to collapse?

2. **Causality Check**: For n=3, write out the full masked score matrix S̃ for arbitrary scores S[i,j]. Show that Z[0] depends only on V[0] after the causal mask.

3. **FlashAttention Savings**: For n=8192, d_k=128, h=32 heads, L=32 layers, BF16 (2 bytes): compute the total memory for the naive attention matrices vs FlashAttention. How many GB does FlashAttention save?

4. **Attention Entropy**: For A[i] = [0.7, 0.2, 0.1], compute H_i. Is this closer to focused (positional) or diffuse (global) attention? What's H_max for 3 positions?
