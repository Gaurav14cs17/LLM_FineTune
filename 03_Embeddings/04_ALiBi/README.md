# Understanding ALiBi (Attention with Linear Biases)

*Zero extra parameters, unlimited length extrapolation — the simplest position encoding*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 The Extrapolation Problem](#11-the-extrapolation-problem)
   - [1.2 ALiBi's Elegant Solution](#12-alibis-elegant-solution)
2. [Mathematical Formulation](#2-mathematical-formulation)
   - [2.1 The Linear Bias](#21-the-linear-bias)
   - [2.2 Multi-Head Slopes](#22-multi-head-slopes)
   - [2.3 Variables Table](#23-variables-table)
3. [Why It Works](#3-why-it-works)
   - [3.1 Softmax Perspective](#31-softmax-perspective)
   - [3.2 Effective Attention Window](#32-effective-attention-window)
4. [Numerical Example](#4-numerical-example)
5. [ALiBi vs RoPE vs Sinusoidal](#5-alibi-vs-rope-vs-sinusoidal)
6. [Common Mistakes](#6-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 The Extrapolation Problem

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Model trained on L=2048, now tested on L=4096     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sinusoidal PE at position 3000:                            │
│    PE(3000) = [sin(3000×ω₀), cos(3000×ω₀), ...]            │
│    Model has NEVER seen this pattern during training!       │
│    → Unpredictable behaviour, quality collapse.             │
│                                                             │
│  Learned PE at position 3000:                               │
│    PE[3000] = ??? (no learned vector exists beyond 2048!)   │
│    → Cannot run at all without extending the table.         │
│                                                             │
│  ALiBi at position 3000:                                    │
│    Just adds -m×|3000-j| to attention score.                │
│    Same formula works at ANY position!                      │
│    → Perfect extrapolation with ZERO modifications.         │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Sinusoidal PE is like memorising specific landmarks on a road (fails on new roads). ALiBi is like knowing "things further away are less relevant" — a universal principle that works everywhere.

### 1.2 ALiBi's Elegant Solution

```
STANDARD ATTENTION:
  score(i,j) = qᵢ · kⱼ / √d_k

ALiBi ATTENTION (add linear distance penalty):
  score(i,j) = qᵢ · kⱼ / √d_k  −  m × |i − j|
                                     ↑ linear penalty for distance!

  NO position encoding added to embeddings!
  NO extra parameters!
  Just subtract a distance-proportional bias from attention scores.
```

---

## 2. Mathematical Formulation

### 2.1 The Linear Bias

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ATTENTION WITH ALiBi:                                       │
│                                                              │
│  A(i,j) = softmax_j[ (qᵢ · kⱼ)/√d_k  +  B(i,j) ]          │
│                                                              │
│  where B(i,j) = −m × (i − j)    for causal attention (j ≤ i)│
│                                                              │
│  BIAS MATRIX B for sequence length 5:                        │
│       j=0    j=1    j=2    j=3    j=4                        │
│  i=0 [ 0   ]                                                │
│  i=1 [-m     0  ]                                            │
│  i=2 [-2m   -m    0  ]                                       │
│  i=3 [-3m   -2m  -m    0  ]                                  │
│  i=4 [-4m   -3m  -2m  -m    0  ]                             │
│                                                              │
│  Tokens further back get LARGER negative bias → less attention│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Multi-Head Slopes

```
DIFFERENT HEADS USE DIFFERENT SLOPES m:

  For H attention heads, slopes are geometric sequence:
    m_h = 2^{-8/H} for h = 1, 2, ..., H

  For H=8 heads:
    m₁ = 2^{-8/8} = 2^{-1} = 1/2
    m₂ = 2^{-16/8} = 2^{-2} = 1/4
    m₃ = 2^{-24/8} = 2^{-3} = 1/8
    m₄ = 2^{-32/8} = 2^{-4} = 1/16
    m₅ = 2^{-40/8} = 2^{-5} = 1/32
    m₆ = 2^{-48/8} = 2^{-6} = 1/64
    m₇ = 2^{-56/8} = 2^{-7} = 1/128
    m₈ = 2^{-64/8} = 2^{-8} = 1/256

EFFECT:
  Head 1 (m=1/2):  strong decay → attends mostly to last ~4 tokens (LOCAL)
  Head 8 (m=1/256): weak decay → attends to last ~500 tokens (GLOBAL)
  
  Different heads naturally specialise in different context ranges!
```

### 2.3 Variables Table

| Symbol | Meaning | Value |
|--------|---------|-------|
| m_h | Slope for head h | 2^{-8h/H} |
| H | Number of attention heads | 8, 32, etc. |
| B(i,j) | Bias at position (i,j) | -m × (i-j) |
| i | Query position | 0 to n-1 |
| j | Key position (j ≤ i for causal) | 0 to i |

---

## 3. Why It Works

### 3.1 Softmax Perspective

```
AFTER SOFTMAX, the attention weight for key at position j:

  α(i,j) ∝ exp(qᵢ·kⱼ/√d_k − m(i−j))
          = exp(qᵢ·kⱼ/√d_k) × exp(−m(i−j))
                ↑ content score        ↑ exponential decay with distance

EFFECTIVE ATTENTION = content_relevance × distance_decay

  For position k steps back:
    decay factor = exp(−m×k)
  
  With m=1/8:
    k=1:   decay = exp(-0.125) = 0.88  (12% reduction)
    k=5:   decay = exp(-0.625) = 0.54  (46% reduction)
    k=10:  decay = exp(-1.25)  = 0.29  (71% reduction)
    k=50:  decay = exp(-6.25)  = 0.002 (99.8% reduction — effectively zero)
```

### 3.2 Effective Attention Window

```
EFFECTIVE WINDOW SIZE (where decay > 5% of peak):
  exp(−m×W) = 0.05  →  W = −ln(0.05)/m = 3/m

  Head 1 (m=1/2):   W = 6 tokens      (very local)
  Head 4 (m=1/16):  W = 48 tokens     (paragraph-level)
  Head 8 (m=1/256): W = 768 tokens    (document-level)

  The model AUTOMATICALLY gets multi-scale attention:
  some heads for local syntax, others for long-range semantics.
  No explicit window design needed!
```

---

## 4. Numerical Example

```
EXAMPLE: H=4 heads, sequence "The cat sat on"

  Slopes: m₁=1/2, m₂=1/4, m₃=1/8, m₄=1/16

  For query at position 3 ("on"), key positions 0,1,2,3:
    Assume content scores q·k/√d = [0.5, 0.8, 1.2, 0.3]
  
  HEAD 1 (m=1/2, LOCAL):
    Biases: [-1.5, -1.0, -0.5, 0]
    Scores: [0.5-1.5, 0.8-1.0, 1.2-0.5, 0.3-0] = [-1.0, -0.2, 0.7, 0.3]
    Softmax: [0.05, 0.12, 0.50, 0.33]
    → Strongly prefers nearby "sat" (pos 2)
  
  HEAD 4 (m=1/16, GLOBAL):
    Biases: [-3/16, -2/16, -1/16, 0] = [-0.19, -0.13, -0.06, 0]
    Scores: [0.31, 0.67, 1.14, 0.30]
    Softmax: [0.12, 0.17, 0.54, 0.17]
    → Also attends to distant "The" (pos 0)
```

---

## 5. ALiBi vs RoPE vs Sinusoidal

| Property | Sinusoidal | RoPE | ALiBi |
|----------|-----------|------|-------|
| Extra parameters | 0 | 0 | 0 |
| Where applied | Added to embeddings | Rotates Q, K | Biases attention scores |
| Encodes | Absolute position | Relative position | Distance penalty |
| Length extrapolation | Poor | Moderate (with PI/YaRN) | Excellent |
| Training efficiency | Good | Good | Good |
| Used in | Original Transformer | LLaMA, Mistral, GPT-NeoX | BLOOM, MPT |
| Quality at train length | Good | Best | Good |
| Quality beyond train length | Degrades fast | Degrades slowly | Maintains well |

---

## 6. Common Mistakes

```
❌ WRONG: ALiBi adds position embeddings to the token representations
✓ RIGHT:  ALiBi adds NO embeddings. It only modifies ATTENTION SCORES
          by subtracting a linear distance penalty. Token embeddings
          are purely content-based (no position info embedded in them).

❌ WRONG: The slope m is learned during training
✓ RIGHT:  Slopes are FIXED hyperparameters (geometric sequence).
          They are set once and never change. This is why ALiBi has
          zero extra parameters compared to the base model.

❌ WRONG: ALiBi can only do local attention
✓ RIGHT:  With small m (like 1/256), the decay is very gentle — tokens
          hundreds of positions back still get meaningful attention weight.
          The multi-head slopes naturally provide BOTH local AND global.

❌ WRONG: ALiBi is strictly better than RoPE
✓ RIGHT:  At the TRAINING length, RoPE typically achieves slightly better
          perplexity. ALiBi's advantage is in EXTRAPOLATION beyond training.
          Most modern LLMs use RoPE (with extension methods) rather than ALiBi.
```

---

## 7. Exercises

1. **Slope Computation**: For H=32 heads, compute slopes m₁, m₈, m₁₆, m₃₂. What is the effective attention window for the first and last head?

2. **Bias Matrix**: For n=6 and m=1/4, write out the full 6×6 bias matrix B. What is the maximum (most negative) bias value?

3. **Extrapolation Test**: A model trained at L=2048 with ALiBi is tested at L=8192. For head with m=1/16, what is the decay factor for a token 8000 positions back? Is it effectively zero?

4. **Comparison**: Compute attention weights (after softmax) for query at pos 10 attending to keys at pos 0-10, with content scores all equal to 1.0. Compare: (a) no position encoding, (b) ALiBi with m=1/8. Show that ALiBi creates a recency bias.
