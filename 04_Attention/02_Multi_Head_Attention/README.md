# Understanding Multi-Head Attention (MHA)

*Why multiple heads, the projection mathematics, and concatenation — from theory to LLaMA-3 numbers*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Multiple Heads?](#11-why-multiple-heads)
   - [1.2 Architecture Diagram](#12-architecture-diagram)
2. [Mathematical Formulation](#2-mathematical-formulation)
   - [2.1 Single Head Recap](#21-single-head-recap)
   - [2.2 Multi-Head Formula](#22-multi-head-formula)
   - [2.3 Variables Table](#23-variables-table)
3. [Parameter Count and FLOPs](#3-parameter-count-and-flops)
   - [3.1 Projection Matrices](#31-projection-matrices)
   - [3.2 FLOPs per Layer](#32-flops-per-layer)
4. [What Heads Learn](#4-what-heads-learn)
   - [4.1 Specialisation Patterns](#41-specialisation-patterns)
5. [Numerical Example](#5-numerical-example)
6. [Common Mistakes](#6-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Multiple Heads?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SINGLE HEAD: one "question" asked of the sequence.         │
│    "Where is the subject of this sentence?"                 │
│                                                             │
│  MULTI-HEAD: multiple "questions" asked IN PARALLEL.        │
│    Head 1: "Where is the subject?"                          │
│    Head 2: "Where is the verb?"                             │
│    Head 3: "What was mentioned recently?"                   │
│    Head 4: "What is the syntactic dependency?"              │
│    ...                                                      │
│                                                             │
│  Each head can attend to DIFFERENT positions                │
│  for DIFFERENT reasons simultaneously!                      │
│                                                             │
│  WITHOUT multi-head: forced to average all "questions" into │
│  one attention pattern → information is lost.               │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Single-head is like asking one person to watch for traffic, pedestrians, AND signs simultaneously. Multi-head is like having 32 spotters, each watching one thing — much more effective.

### 1.2 Architecture Diagram

```
Input x ∈ ℝ^{n×d}
    │
    ├──── W_Q ────→ Q ∈ ℝ^{n×d} ──→ split into h heads ──→ Q₁,Q₂,...,Q_h
    ├──── W_K ────→ K ∈ ℝ^{n×d} ──→ split into h heads ──→ K₁,K₂,...,K_h
    └──── W_V ────→ V ∈ ℝ^{n×d} ──→ split into h heads ──→ V₁,V₂,...,V_h
                                         │
                                ┌────────┼────────┐
                                ↓        ↓        ↓
                          Attention₁ Attention₂ ... Attention_h
                          (n×d_k)    (n×d_k)       (n×d_k)
                                ↓        ↓        ↓
                                └────────┼────────┘
                                         ↓ concatenate
                              [head₁ ; head₂ ; ... ; head_h] ∈ ℝ^{n×d}
                                         │
                                    W_O ──→ output ∈ ℝ^{n×d}
```

---

## 2. Mathematical Formulation

### 2.1 Single Head Recap

```
Single head attention (d_k dimensional):
  head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i K_iᵀ / √d_k) × V_i
  
  Q_i = x × W_Q^{(i)} ∈ ℝ^{n×d_k}
  K_i = x × W_K^{(i)} ∈ ℝ^{n×d_k}
  V_i = x × W_V^{(i)} ∈ ℝ^{n×d_v}
```

### 2.2 Multi-Head Formula

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  MultiHead(x) = Concat(head₁, ..., head_h) × W_O            │
│                                                              │
│  where head_i = softmax(Q_i K_iᵀ / √d_k) × V_i             │
│                                                              │
│  Dimensions:                                                 │
│    d_k = d_v = d / h    (split dimension equally per head)   │
│    W_Q^{(i)} ∈ ℝ^{d × d_k}                                  │
│    W_K^{(i)} ∈ ℝ^{d × d_k}                                  │
│    W_V^{(i)} ∈ ℝ^{d × d_v}                                  │
│    W_O ∈ ℝ^{(h×d_v) × d} = ℝ^{d × d}                       │
│                                                              │
│  EQUIVALENTLY (how it's actually implemented):               │
│    W_Q ∈ ℝ^{d × d}  (all heads packed into one big matrix)  │
│    Q = x × W_Q  →  split into [Q₁; Q₂; ...; Q_h]           │
│    (split along last dimension into h chunks of size d_k)    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 Variables Table

| Symbol | Meaning | LLaMA-3 8B |
|--------|---------|-----------|
| d | Model dimension | 4096 |
| h | Number of attention heads | 32 |
| d_k | Key/query dimension per head = d/h | 128 |
| d_v | Value dimension per head = d/h | 128 |
| W_Q, W_K, W_V | Projection matrices | ℝ^{4096×4096} each |
| W_O | Output projection | ℝ^{4096×4096} |
| n | Sequence length | 8192 |

---

## 3. Parameter Count and FLOPs

### 3.1 Projection Matrices

```
PARAMETER COUNT per attention layer:
  W_Q: d × d = 4096² = 16.8M
  W_K: d × d_k × h_kv = 4096 × 128 × 8 = 4.2M  (GQA in LLaMA-3)
  W_V: d × d_v × h_kv = 4096 × 128 × 8 = 4.2M  (GQA in LLaMA-3)
  W_O: d × d = 4096² = 16.8M
  
  TOTAL (standard MHA): 4 × d² = 4 × 4096² = 67.1M per layer
  TOTAL (GQA, h_kv=8):  d²(1 + 2×h_kv/h + 1) = ~42M per layer
  
  × 32 layers = 1.34B params in attention (standard MHA)
```

### 3.2 FLOPs per Layer

```
FLOPs per token per attention layer (forward pass):

  Q projection: 2 × n × d × d = 2 × d² per token
  K projection: 2 × n × d × d_kv = 2 × d × d_kv per token
  V projection: same as K
  QKᵀ scores:   2 × n × d_k × h = 2 × n × d per token (amortised)
  Score × V:    2 × n × d_v × h = 2 × n × d per token (amortised)
  O projection: 2 × d × d per token

  TOTAL ≈ 8 × d² per token (for standard MHA where all dims = d)
  For d=4096: 8 × 4096² = 134M FLOPs per token per layer
```

---

## 4. What Heads Learn

### 4.1 Specialisation Patterns

```
OBSERVED HEAD SPECIALISATIONS (from analysis of trained models):

┌─────────────────────────────────────────────────────────────┐
│  Head Type          │ Attention Pattern                      │
├─────────────────────┼───────────────────────────────────────┤
│  Positional heads   │ Attend to fixed relative positions    │
│                     │ (e.g., always attend 1 position back) │
│  Induction heads    │ Copy patterns: "A B ... A" → attend B│
│                     │ (key for in-context learning!)        │
│  Syntactic heads    │ Attend to syntactic dependencies     │
│                     │ (verb → subject, adj → noun)         │
│  Rare token heads   │ Attend strongly to delimiter tokens  │
│                     │ ([BOS], ".", "\n")                    │
│  Aggregate heads    │ Near-uniform attention (averaging)    │
│                     │ (broad context summary)               │
└─────────────────────┴───────────────────────────────────────┘

NOT ALL HEADS ARE EQUALLY IMPORTANT:
  ~20% of heads can be pruned with < 1% quality loss.
  "Induction heads" in layers 2-4 are CRITICAL for ICL.
```

---

## 5. Numerical Example

```
EXAMPLE: d=4, h=2, d_k=d_v=2, n=3

  Input x = [[1,0,1,0], [0,1,0,1], [1,1,0,0]]  (3 tokens × 4 dims)

  W_Q = [[1,0,0,1], [0,1,1,0], [1,0,0,1], [0,1,1,0]]  (4×4)
  Q = x × W_Q → split into Q₁(3×2), Q₂(3×2)

  Head 1 (first 2 dims):
    Q₁ = [[1,0],[0,1],[1,0]]  K₁ = [[1,1],[0,0],[1,1]]  V₁ = [[1,0],[0,1],[1,1]]
    Scores = Q₁×K₁ᵀ/√2 = [[0.71,0,0.71],[0.71,0,0.71],[0.71,0,0.71]]
    After causal mask + softmax → attention weights
    head₁ = weights × V₁ ∈ ℝ^{3×2}

  Head 2 (last 2 dims):
    Q₂, K₂, V₂ from other half of projections
    Similar computation → head₂ ∈ ℝ^{3×2}

  Concatenate: [head₁ ; head₂] ∈ ℝ^{3×4}
  Output = [head₁ ; head₂] × W_O ∈ ℝ^{3×4}
```

---

## 6. Common Mistakes

```
❌ WRONG: More heads always means better quality
✓ RIGHT:  There's a sweet spot. Too many heads → d_k becomes very small
          → each head has limited representational capacity.
          LLaMA-3 uses h=32 with d_k=128. Going to h=128 (d_k=32) hurts.

❌ WRONG: Each head operates on a different subset of input features
✓ RIGHT:  Each head has its OWN projection matrices W_Q, W_K, W_V.
          All heads see the FULL input x, but project it differently.
          The split is in the projected space, not the input space.

❌ WRONG: Multi-head attention has more parameters than single-head
✓ RIGHT:  Parameter count is the SAME! With h heads of size d/h each:
          Total W_Q params = h × (d × d/h) = d²  (same as single d×d).
          Multi-head just splits the same parameter budget differently.

❌ WRONG: The output projection W_O is optional
✓ RIGHT:  W_O is essential. Without it, the output is just concatenated
          independent head outputs — no cross-head interaction.
          W_O allows the model to MIX information across heads.
```

---

## 7. Exercises

1. **Dimension Check**: For d=2048, h=16: compute d_k, d_v. Write the shapes of W_Q, W_K, W_V, W_O. Verify that Concat(head₁,...,head_h) has shape (n, d).

2. **FLOPs**: For d=4096, h=32, n=2048: compute total FLOPs for one MHA layer (Q,K,V projections + attention scores + output projection). What fraction of total transformer layer FLOPs is attention vs FFN?

3. **Head Pruning**: If you remove 8 out of 32 heads: how do parameter counts change? How do FLOPs change? (Assume you resize W_O to match.)

4. **Single vs Multi**: Prove that single-head attention with d_k=d (same total dimension) requires the same number of parameters as multi-head with h heads of d/h each.
