# Understanding Token Embeddings

*From integer token IDs to dense vectors — the mathematical foundation of neural text representation*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What Are Token Embeddings?](#11-what-are-token-embeddings)
   - [1.2 Pipeline Position](#12-pipeline-position)
2. [Formal Definition](#2-formal-definition)
   - [2.1 The Embedding Matrix](#21-the-embedding-matrix)
   - [2.2 Lookup as Matrix Multiplication](#22-lookup-as-matrix-multiplication)
   - [2.3 Variables Table](#23-variables-table)
3. [Geometric Interpretation](#3-geometric-interpretation)
   - [3.1 Semantic Similarity](#31-semantic-similarity)
   - [3.2 Cosine Distance Meaning](#32-cosine-distance-meaning)
4. [Initialisation](#4-initialisation)
   - [4.1 Why Random Initialisation Matters](#41-why-random-initialisation-matters)
   - [4.2 Xavier/He Initialisation for Embeddings](#42-xavierhe-initialisation-for-embeddings)
5. [Weight Tying (Shared Embeddings)](#5-weight-tying-shared-embeddings)
   - [5.1 The Parameter Saving](#51-the-parameter-saving)
   - [5.2 Mathematical Justification](#52-mathematical-justification)
6. [Numerical Example](#6-numerical-example)
7. [Common Mistakes](#7-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 What Are Token Embeddings?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Token ID = 42 means nothing geometrically.                 │
│  Token ID = 43 is not "similar" to 42.                      │
│                                                             │
│  Embedding = learned vector that ENCODES MEANING.           │
│  embed("cat") ≈ embed("kitten")  (similar meaning → close) │
│  embed("cat") ≠ embed("bicycle")  (different → far apart)  │
│                                                             │
│  ┌─────────────┐        ┌────────────────────────────┐      │
│  │ token_id=42 │──────► │ [0.12, -0.34, 0.89, ...]  │      │
│  │ (integer)   │ lookup │ d-dimensional real vector  │      │
│  └─────────────┘        └────────────────────────────┘      │
│                                                             │
│  The entire embedding matrix is LEARNED during training.    │
│  There is no formula — it's pure gradient descent.          │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Token IDs are like house addresses (arbitrary numbers). Embeddings are like GPS coordinates — they place each "house" (word) in a continuous space where nearby points are semantically related.

### 1.2 Pipeline Position

```
Raw text: "The cat sat"
    ↓ Tokenizer
Token IDs: [464, 2368, 3332]
    ↓ EMBEDDING LAYER ← THIS TOPIC
Dense vectors: [[0.12,-0.34,...], [0.56,0.78,...], [0.23,-0.11,...]]
    ↓ (+ Position Encoding)
    ↓ Transformer layers
Logits → next token prediction
```

---

## 2. Formal Definition

### 2.1 The Embedding Matrix

```
┌──────────────────────────────────────────────────────────────┐
│  EMBEDDING MATRIX E ∈ ℝ^{|V| × d}                           │
│                                                              │
│  |V| = vocabulary size (e.g., 128,256 for LLaMA-3)          │
│  d   = embedding dimension (e.g., 4096 for LLaMA-3 8B)      │
│                                                              │
│  E = ┌ e₀ ─────────────────── ┐  ← row 0: embed("the")     │
│      │ e₁ ─────────────────── │  ← row 1: embed("a")       │
│      │ e₂ ─────────────────── │  ← row 2: embed("cat")     │
│      │ ⋮                       │                             │
│      └ e_{|V|-1} ────────────  ┘  ← row |V|-1: last token   │
│                                                              │
│  Each row eₜ ∈ ℝᵈ is a d-dimensional learned vector.        │
│                                                              │
│  PARAMETER COUNT:                                            │
│    |V| × d = 128,256 × 4,096 = 525 million parameters!     │
│    ≈ 6.5% of total LLaMA-3 8B parameters                   │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Lookup as Matrix Multiplication

```
FORWARD PASS (lookup):
  Input:  token ID t ∈ {0, 1, ..., |V|-1}
  Output: x_t = E[t, :] ∈ ℝᵈ

EQUIVALENT to one-hot multiplication:
  one_hot(t) ∈ {0,1}^{|V|}  (1 at position t, 0 elsewhere)
  x_t = one_hot(t)ᵀ × E = E[t, :]

  But IMPLEMENTED as direct index lookup → O(d) not O(|V|×d)

BACKWARD PASS (gradient):
  ∂L/∂E[t, :] = ∂L/∂x_t   (gradient only updates row t)
  ∂L/∂E[j, :] = 0  for j ≠ t  (other rows untouched)

  → SPARSE gradient: only embeddings of tokens IN THE BATCH get updated.
  → Rare tokens learn slowly (few gradient updates per epoch).
```

### 2.3 Variables Table

| Symbol | Meaning | LLaMA-3 8B value |
|--------|---------|-----------------|
| E | Embedding matrix | ℝ^{128256 × 4096} |
| \|V\| | Vocabulary size | 128,256 |
| d | Embedding dimension | 4,096 |
| e_t | Embedding vector for token t | ℝ^{4096} |
| one_hot(t) | One-hot vector for token t | ℝ^{128256} |

---

## 3. Geometric Interpretation

### 3.1 Semantic Similarity

```
AFTER TRAINING, embeddings cluster by meaning:

  2D projection (PCA) of embedding space:
  
       "queen" ●
                    "king" ●
  "woman" ●                    ← royalty cluster
                    "man" ●
  
  
              "car" ●
  "bicycle" ●          "truck" ●    ← vehicle cluster
  
  
  FAMOUS RELATIONSHIP (Word2Vec, also emerges in LLMs):
    embed("king") - embed("man") + embed("woman") ≈ embed("queen")
    
    The vector difference encodes the CONCEPT "royalty" independent of gender.
```

### 3.2 Cosine Distance Meaning

```
COSINE SIMILARITY between embeddings:
  cos(eᵢ, eⱼ) = eᵢ · eⱼ / (‖eᵢ‖ × ‖eⱼ‖)

  cos ≈ 1:   tokens are semantically similar ("cat" ↔ "kitten")
  cos ≈ 0:   tokens are unrelated ("cat" ↔ "finance")
  cos ≈ -1:  tokens are semantic opposites ("hot" ↔ "cold")

IN PRACTICE (after training):
  cos("dog", "puppy") ≈ 0.85
  cos("dog", "canine") ≈ 0.78
  cos("dog", "table") ≈ 0.05
  cos("good", "bad") ≈ -0.15  (opposite but same domain)
```

---

## 4. Initialisation

### 4.1 Why Random Initialisation Matters

```
┌─────────────────────────────────────────────────────────────┐
│  IF ALL EMBEDDINGS INITIALIZED TO SAME VECTOR:              │
│    All tokens produce identical hidden states!              │
│    Gradients are identical → symmetry never breaks!         │
│    Model CANNOT learn to distinguish tokens.               │
│                                                             │
│  IF INITIALIZED WITH TOO LARGE VALUES:                      │
│    Hidden states have large magnitude at start.             │
│    Softmax saturates → gradient vanishes.                   │
│    Training diverges or converges very slowly.              │
│                                                             │
│  IF INITIALIZED WITH TOO SMALL VALUES:                      │
│    Hidden states ≈ 0 → all information lost.                │
│    Gradients are tiny → training is extremely slow.         │
│                                                             │
│  GOLDILOCKS: small random values with controlled variance.  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Xavier/He Initialisation for Embeddings

```
STANDARD INITIALISATION:
  E[t, :] ~ N(0, σ²)    where σ = 1/√d

  For LLaMA-3 8B (d=4096):
    σ = 1/√4096 = 1/64 = 0.0156
    Each embedding element ∈ [-0.05, 0.05] (roughly, within 3σ)

WHY 1/√d?
  If x = E[t, :] has entries ~ N(0, σ²):
    ‖x‖² = Σᵢ xᵢ² ≈ d × σ²   (by LLN)
    
  We want ‖x‖ ≈ 1 (unit-scale hidden states):
    d × σ² = 1  →  σ = 1/√d  ✓

  Alternative: uniform U(-a, a) where a = √(3/d)
    Var[U(-a,a)] = a²/3 = 1/d  → same variance. ✓
```

---

## 5. Weight Tying (Shared Embeddings)

### 5.1 The Parameter Saving

```
┌─────────────────────────────────────────────────────────────┐
│  WITHOUT WEIGHT TYING:                                      │
│    Input embedding:  E_in ∈ ℝ^{|V| × d}   (525M params)   │
│    Output head:      W_out ∈ ℝ^{d × |V|}  (525M params)    │
│    Total:            1050M params just for embed+head!      │
│                                                             │
│  WITH WEIGHT TYING (LLaMA-3 uses this):                     │
│    E_in = W_outᵀ    (shared matrix!)                        │
│    Total:            525M params (50% savings!)             │
│                                                             │
│  For LLaMA-3 8B: saves 525M / 8030M = 6.5% of parameters  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Mathematical Justification

```
OUTPUT LOGIT for token v:
  logit_v = hidden_state · w_out_v

WITH WEIGHT TYING:
  logit_v = hidden_state · e_v   (dot product with input embedding)

INTUITION:
  "The probability of outputting token v is proportional to how similar 
   the final hidden state is to the EMBEDDING of token v."

  P(next = v | context) ∝ exp(h · e_v)
  
  This makes sense: if the model's internal representation is "cat-like",
  it should output tokens whose embeddings are similar to that state.

GRADIENT BENEFIT:
  With tying: embedding gets gradient from BOTH:
    1. Forward pass (as input embedding)
    2. Loss computation (as output classifier)
  → Rare tokens learn faster (get gradient even when not in input).
```

---

## 6. Numerical Example

```
EXAMPLE: LLaMA-3 8B, token "hello" has ID=9707

  Embedding lookup:
    x = E[9707, :]  ∈ ℝ^{4096}
    x = [0.0123, -0.0089, 0.0234, -0.0156, ..., 0.0067]  (4096 values)
  
  Norm: ‖x‖ = √(Σᵢ xᵢ²) ≈ 1.0 (after training)
  
  Memory for one embedding: 4096 × 2 bytes (BF16) = 8 KB
  Memory for full E matrix: 128,256 × 4096 × 2 = 1.0 GB (BF16)
  
  For a batch of tokens [9707, 278, 526]:
    X = E[[9707, 278, 526], :]  ∈ ℝ^{3 × 4096}
    (parallel lookup of 3 embeddings)

COMPUTATION COST:
  Forward: O(batch_size × seq_len × d) = essentially free (just memory access)
  Backward: sparse update to only seen token rows
  
  Embedding lookup is NEVER the bottleneck — attention/FFN dominate FLOPs.
```

---

## 7. Common Mistakes

```
❌ WRONG: Embeddings are fixed mathematical functions (like sin/cos PE)
✓ RIGHT:  Token embeddings are 100% LEARNED from data. They start random
          and gradually organise into semantic clusters through gradient descent.
          Only position encodings can be fixed (sinusoidal) or learned.

❌ WRONG: Token ID proximity means semantic proximity
✓ RIGHT:  Token IDs are ARBITRARY integers assigned by the tokenizer.
          Token 42 and 43 have NO inherent similarity.
          Only the LEARNED embedding vectors encode semantic relationships.

❌ WRONG: Each word gets one embedding
✓ RIGHT:  Each TOKEN (subword unit) gets one embedding.
          "unhappiness" might be ["un", "happiness"] → 2 embeddings looked up.
          The word-level meaning emerges from context processing in later layers.

❌ WRONG: The embedding matrix is the most computationally expensive part
✓ RIGHT:  The embedding lookup is O(n×d) — essentially free.
          Attention is O(n²×d) and FFN is O(n×d×4d) — much more expensive.
          But the embedding matrix is large in PARAMETERS (6.5% of total).
```

---

## 8. Exercises

1. **Parameter Count**: For a model with |V|=50,257 (GPT-2) and d=768: compute the embedding matrix size in parameters and in MB (FP32). What fraction of GPT-2's 124M total parameters is this?

2. **Initialisation Variance**: With σ=1/√d and d=4096, what is the expected L2 norm of a randomly initialised embedding vector? What if you accidentally use σ=1 instead?

3. **Weight Tying Savings**: Model has |V|=128,256 and d=4096. Compare total parameter count with and without weight tying for the embedding + output head. Express savings as percentage of an 8B model.

4. **Sparse Gradients**: In a batch of 2048 tokens from a 128K vocabulary, what fraction of embedding rows receive a gradient update? If you train for 1M steps with batch=2048, how many gradient updates does a token appearing with frequency 0.001% receive?
