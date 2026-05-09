# Understanding RoPE: Rotary Position Embedding

*A comprehensive guide from design goal to implementation and context extension*

---

**RoPE (Rotary Position Embedding)** is the position encoding used in LLaMA, PaLM, GPT-NeoX, and most modern LLMs. Unlike sinusoidal PE (added to token embeddings) or learned PE (a lookup table), RoPE **rotates** the query and key vectors before the attention dot product — encoding relative position directly into the attention score.

This guide proves why RoPE works from first principles.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Position Encoding?](#11-why-position-encoding)
   - [1.2 What Makes RoPE Different](#12-what-makes-rope-different)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The Design Goal](#2-the-design-goal)
   - [2.1 What We Want](#21-what-we-want)
   - [2.2 The 2D Rotation Solution](#22-the-2d-rotation-solution)
3. [Full d-Dimensional RoPE](#3-full-d-dimensional-rope)
   - [3.1 Block-Diagonal Rotation](#31-block-diagonal-rotation)
   - [3.2 Frequency Design](#32-frequency-design)
4. [Efficient Implementation](#4-efficient-implementation)
   - [4.1 The Rotate-Half Trick](#41-the-rotate-half-trick)
   - [4.2 Numerical Example](#42-numerical-example)
5. [Context Extension](#5-context-extension)
   - [5.1 The Extrapolation Problem](#51-the-extrapolation-problem)
   - [5.2 Position Interpolation](#52-position-interpolation)
   - [5.3 NTK-Aware Scaling](#53-ntk-aware-scaling)
6. [Summary](#6-summary)
   - [6.1 Formulas Quick Reference](#61-formulas-quick-reference)
   - [6.2 Comparison Table](#62-comparison-table)
   - [6.3 Common Mistakes](#63-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Position Encoding?

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Transformer attention is permutation-equivariant   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "The cat sat on the mat"                                   │
│  "mat the on sat cat The"                                   │
│                                                             │
│  Without position encoding: both produce IDENTICAL outputs  │
│  (just in different order — the model can't distinguish!)   │
│                                                             │
│  PROOF: S = QKᵀ/√d = (XW_Q)(XW_K)ᵀ/√d                     │
│  Permuting X → permutes S → permutes output Z               │
│  Same content, different order → same attention patterns     │
│                                                             │
│  POSITION ENCODING FIX:                                     │
│  Make each token's representation depend on its position.   │
│  Then "cat" at position 2 and "cat" at position 5 look     │
│  different to the attention mechanism.                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 What Makes RoPE Different

```
SINUSOIDAL PE (original Transformer):
  x̃_t = E[w_t] + PE(t)      ← ADD position vector to embedding

LEARNED PE (BERT, GPT-2):
  x̃_t = E[w_t] + P[t]       ← LOOKUP a learnable position vector

RoPE:
  NO modification to the input embeddings at all!
  Instead, ROTATE Q and K vectors right before the dot product:
  
  score(q_t, k_s) = (R_t q)ᵀ (R_s k) = qᵀ R_tᵀ R_s k = qᵀ R_{s-t} k
                    ↑ RoPE is applied HERE (inside attention)
  
  The attention score depends only on q, k, and (s-t) — RELATIVE position!
```

> **Real-World Analogy**: Instead of stamping a "position number" on each token before it enters the room, RoPE tilts the coordinate system differently for each position. When two tokens "look at" each other, the tilt difference encodes their relative distance.

### 1.3 Pipeline Summary

```
INPUT: query q ∈ ℝᵈ at position t, key k ∈ ℝᵈ at position s
        ↓
Step 1: Pair up dimensions into d/2 pairs: (q₁,q₂), (q₃,q₄), ...
        ↓
Step 2: For pair i, rotate by angle t×θᵢ (using 2D rotation matrix)
        q_rotated[2i]   = q[2i]cos(tθᵢ) − q[2i+1]sin(tθᵢ)
        q_rotated[2i+1] = q[2i]sin(tθᵢ) + q[2i+1]cos(tθᵢ)
        ↓
Step 3: Similarly rotate k by angle s×θᵢ
        ↓
Step 4: Compute attention score:
        score = q_rotated · k_rotated
              = qᵀ R_{s-t} k   (depends only on relative position s-t!)
```

---

## 2. The Design Goal

### 2.1 What We Want

```
DESIGN GOAL:

  ⟨f(q, t_q), f(k, t_k)⟩ = g(q, k, t_k − t_q)

  where:
    f: ℝᵈ × ℤ → ℝᵈ  encodes a vector WITH its position
    g: ℝᵈ × ℝᵈ × ℤ → ℝ depends only on RELATIVE offset

WHY RELATIVE?
  The phrase "cat sat" has the same syntactic relationship
  whether it appears at positions (1,2) or (1000,1001).
  The model should treat them identically!

  ABSOLUTE position PE (sinusoidal, learned):
    "cat" at position 1 ≠ "cat" at position 1000
    → model must learn positional generalization separately

  RELATIVE position encoding (RoPE):
    score("cat"@t, "sat"@t+1) = score("cat"@1000, "sat"@1001)
    → generalisation is built in by design ✓
```

### 2.2 The 2D Rotation Solution

```
IN 2D: set f(x, t) = R(tθ) × x   (rotate the 2D vector x by angle tθ)

  R(α) = ┌ cos α  −sin α ┐   (standard 2D rotation matrix)
          └ sin α   cos α ┘

VERIFY THE DESIGN GOAL:

  ⟨f(q, t_q), f(k, t_k)⟩
  = ⟨R(t_q θ) q, R(t_k θ) k⟩
  = qᵀ R(t_q θ)ᵀ R(t_k θ) k

  Key property: R(α)ᵀ = R(−α)  and  R(−α)R(β) = R(β−α)

  = qᵀ R(−t_q θ) R(t_k θ) k
  = qᵀ R((t_k − t_q) θ) k
  = g(q, k, t_k − t_q)   ✓  depends only on relative offset!

PROOF that R(α)R(β) = R(α+β):
  ┌ cos α  −sin α ┐ ┌ cos β  −sin β ┐
  │ sin α   cos α ┘ └ sin β   cos β ┘
  = ┌ cos α cos β − sin α sin β    −cos α sin β − sin α cos β ┐
    └ sin α cos β + cos α sin β    −sin α sin β + cos α cos β ┘
  = ┌ cos(α+β)   −sin(α+β) ┐   (using angle addition formulas)
    └ sin(α+β)    cos(α+β) ┘
  = R(α+β)   ■
```

---

## 3. Full d-Dimensional RoPE

### 3.1 Block-Diagonal Rotation

```
EXTENSION to d dimensions:
  Pair up dimensions: (x₁,x₂), (x₃,x₄), ..., (x_{d-1}, x_d)
  Apply a DIFFERENT rotation angle to each pair.

BLOCK-DIAGONAL ROTATION MATRIX Rθ,t ∈ ℝᵈˣᵈ:

  ┌ R(tθ₀)                              ┐
  │         R(tθ₁)                      │
  │                  R(tθ₂)             │
  │                           ⋱         │
  └                              R(tθ_{d/2-1}) ┘

  where each R(tθᵢ) = ┌ cos(tθᵢ)  −sin(tθᵢ) ┐  is a 2×2 block
                       └ sin(tθᵢ)   cos(tθᵢ) ┘

ATTENTION SCORE with RoPE:
  score(q@t_q, k@t_k) = (Rθ,t_q q)ᵀ (Rθ,t_k k)
                       = qᵀ Rθ,t_qᵀ Rθ,t_k k
                       = qᵀ Rθ,(t_k−t_q) k   ← relative position only! ✓
```

### 3.2 Frequency Design

```
FREQUENCIES (identical to sinusoidal PE):

  θᵢ = 1 / 10000^{2i/d}   for i = 0, 1, ..., d/2 − 1

  ┌─────────────────────────────────────────────────────────────┐
  │  FREQUENCY SPECTRUM:                                        │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  Pair index i   Frequency θᵢ      Cycle length 2π/θᵢ       │
  │  ─────────────────────────────────────────────────────────  │
  │  i = 0          θ₀ = 1.0          ~6 positions             │
  │  i = 8          θ₈ = 0.0025       ~2500 positions           │
  │  i = 32         θ₃₂ = 1×10⁻⁴     ~63000 positions           │
  │  i = d/2-1      θ_min≈10⁻⁴       ~62800 positions           │
  │                                                             │
  │  HIGH-FREQ pairs (small i):                                 │
  │    Rotate quickly → sensitive to nearby positions           │
  │    Used for LOCAL syntax (subject-verb distance)            │
  │                                                             │
  │  LOW-FREQ pairs (large i):                                  │
  │    Rotate slowly → sensitive to global structure            │
  │    Used for LONG-RANGE dependencies (document structure)    │
  └─────────────────────────────────────────────────────────────┘
```

---

## 4. Efficient Implementation

### 4.1 The Rotate-Half Trick

```
NAIVE: build the full d×d rotation matrix (O(d²) memory, O(d²) compute)

EFFICIENT: use the rotate-half trick (O(d) memory, O(d) compute)

ROTATE-HALF operation:
  x = [x₁, x₂, x₃, x₄, ..., x_{d-1}, x_d]
  rotate_half(x) = [−x_{d/2+1}, ..., −x_d, x₁, ..., x_{d/2}]
                    ↑ negate second half     ↑ keep first half

APPLYING RoPE:
  cos_t = [cos(tθ₀), cos(tθ₁), ..., cos(tθ_{d/2-1}),   <- first half
           cos(tθ₀), cos(tθ₁), ..., cos(tθ_{d/2-1})]   <- repeated

  sin_t = [sin(tθ₀), sin(tθ₁), ..., sin(tθ_{d/2-1}),
           sin(tθ₀), sin(tθ₁), ..., sin(tθ_{d/2-1})]

  Rθ,t × x = x ⊙ cos_t + rotate_half(x) ⊙ sin_t

OPERATIONS: just 2 element-wise multiplications and 1 addition!
```

### 4.2 Numerical Example

```
d = 4, t = 2, θ₀ = 1.0, θ₁ = 0.01

x = [a, b, c, d]

Step 1: rotate_half(x) = [−c, −d, a, b]

Step 2: cos_t = [cos(2×1.0), cos(2×0.01), cos(2×1.0), cos(2×0.01)]
               = [cos(2), cos(0.02), cos(2), cos(0.02)]
               = [−0.416, 0.9998, −0.416, 0.9998]

Step 3: sin_t = [sin(2), sin(0.02), sin(2), sin(0.02)]
               = [0.909, 0.020, 0.909, 0.020]

Step 4: result = x ⊙ cos_t + rotate_half(x) ⊙ sin_t
       = [a×(−0.416), b×0.9998, c×(−0.416), d×0.9998]
       + [−c×0.909, −d×0.020, a×0.909, b×0.020]
       = [−0.416a − 0.909c,   0.9998b − 0.020d,
           0.909a − 0.416c,   0.9998d + 0.020b]

CHECK: this matches R(tθ₀)×[a,c] and R(tθ₁)×[b,d] exactly ✓
```

---

## 5. Context Extension

### 5.1 The Extrapolation Problem

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Model trained with max context L_train            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  At position t = L_train + 1:                               │
│  → The rotation angle tθᵢ is OUTSIDE the training range    │
│  → The model has never seen this angle during training      │
│  → Attention patterns become erratic, quality drops!       │
│                                                             │
│  EXAMPLE: LLaMA-3 8B trained with L_train = 8192            │
│  At position 8193: model behaves unpredictably              │
│                                                             │
│  HOW BAD IS IT?                                             │
│  For high-frequency pairs (i=0, θ₀=1):                     │
│    Trained on tθ₀ ∈ [0, 8192]                              │
│    Position 8193: tθ₀ = 8193 (many full rotations — OK)    │
│                                                             │
│  For low-frequency pairs (i=d/2-1, θ_min≈10⁻⁴):            │
│    Trained on tθ_min ∈ [0, 0.82]  (< 1 full rotation)      │
│    Position 16384: tθ_min = 1.64  (never seen!)            │
│    → These pairs extrapolate badly!                        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Position Interpolation

```
IDEA (Chen et al. 2023): Compress all positions to fit within training range.

For target length L_target, scale position t → t × L_train / L_target:

  RoPE(x, t)  →  RoPE(x, t × L_train / L_target)

EXAMPLE (L_train=4096, L_target=8192):
  Position 8191 → treated as position 4095.5
  All angles stay within [0, tθᵢ_max_trained] ✓

┌─────────────────────────────────────────────────────────────┐
│  BEFORE:                                                    │
│  Positions: 0  1  2  3  ... 4095  4096  4097 ... 8191      │
│  Angles:    0  θ  2θ  ...  4095θ [UNSEEN TERRITORY!!!]      │
│                                                             │
│  AFTER INTERPOLATION:                                       │
│  Position t → angle (t/2)×θ                                 │
│  All positions stay within trained angular range ✓          │
└─────────────────────────────────────────────────────────────┘

COST: needs 1000+ fine-tuning steps to adapt to "denser" angles.
```

### 5.3 NTK-Aware Scaling

```
BETTER APPROACH (kaiokendev 2023): change the frequency BASE.

ORIGINAL: θᵢ = 1 / 10000^{2i/d}   (base = 10000)

NTK-SCALED: θᵢ' = 1 / (b')^{2i/d}

  where b' = b × (L_target / L_train)^{d/(d-2)}

EFFECT:
  HIGH-FREQ pairs (small i): θᵢ' ≈ θᵢ × (L_train/L_target)  ← interpolated
  LOW-FREQ pairs (large i):  θᵢ' ≈ θᵢ                         ← preserved

INTUITION:
  ┌─────────────────────────────────────────────────────────────┐
  │  HIGH-FREQ (local syntax): compress to fit in trained range │
  │    Nearby tokens still interact the same way ✓              │
  │                                                             │
  │  LOW-FREQ (global structure): keep as-is, extrapolate        │
  │    Long-range patterns are similar at any context length ✓  │
  └─────────────────────────────────────────────────────────────┘

ADVANTAGE: works WITHOUT fine-tuning for 2-4× context extension!
LLaMA-3 uses 128K context with YaRN (an improved version of NTK scaling).
```

---

## 6. Summary

### 6.1 Formulas Quick Reference

**RoPE Rotation:**

```
For pair i at position t:
  q'[2i]   = q[2i]cos(tθᵢ) − q[2i+1]sin(tθᵢ)
  q'[2i+1] = q[2i]sin(tθᵢ) + q[2i+1]cos(tθᵢ)
```

**Frequencies:**

```
θᵢ = 1 / 10000^{2i/d}   for i = 0, 1, ..., d/2 − 1
```

**Attention Score (relative position):**

```
score(q@t, k@s) = qᵀ Rθ,(s−t) k   (depends only on s−t)
```

**Efficient Implementation:**

```
RoPE(x, t) = x ⊙ cos_t + rotate_half(x) ⊙ sin_t
```

**NTK-Aware Base Scaling:**

```
b' = b × (L_target / L_train)^{d/(d-2)}
```

### 6.2 Comparison Table

| Method | Extra Params | Extrapolates | Relative Pos | Performance |
|--------|-------------|-------------|--------------|-------------|
| Sinusoidal PE | 0 | ✓ (partial) | Partly | Baseline |
| Learned PE | L_max × d | ❌ | ❌ | +0.5% |
| ALiBi | 0 | ✓ | ✓ | +1-2% |
| **RoPE** | **0** | **✓ (w/ext)** | **✓** | **+1-3%** |

### 6.3 Common Mistakes

```
❌ WRONG: RoPE is added to the token embeddings
✓ RIGHT:  RoPE rotates Q and K inside attention. The token embeddings
          X are unchanged. RoPE is applied: Q' = RoPE(Q), K' = RoPE(K)

❌ WRONG: All frequencies rotate at the same speed
✓ RIGHT:  θᵢ decreases with i. Low-index pairs rotate fast (local),
          high-index pairs rotate slowly (global structure).

❌ WRONG: You need to materialise the d×d rotation matrix
✓ RIGHT:  Use the rotate-half trick: RoPE(x,t) = x⊙cos_t + rot_half(x)⊙sin_t
          This is O(d) not O(d²) and what PyTorch/vLLM/llama.cpp all use.

❌ WRONG: Position interpolation for context extension works out of the box
✓ RIGHT:  Position interpolation needs 1000+ fine-tuning steps to adapt.
          NTK-aware scaling works WITHOUT fine-tuning for 2-4× extension.
```

---

## 7. Exercises

1. **Rotation Proof**: For d=4 and t=3, write out the full 4×4 rotation matrix Rθ,3 with θ₀=1, θ₁=0.01. Verify that Rθ,3ᵀ × Rθ,3 = I (it's orthogonal).

2. **Design Goal Verification**: For the 2D case with θ=0.5, compute score(q@2, k@5) and score(q@7, k@10) using vectors q=[1,0] and k=[0,1]. Show they are equal (same relative offset = 3).

3. **NTK Scaling**: For d=4096, L_train=4096, L_target=32768: compute b'. How much does the lowest frequency θ_{d/2-1} change? Does the highest frequency θ₀ change much?

4. **Context Extension Comparison**: LLaMA-3 uses 128K context with YaRN. If L_train=8192 and L_target=131072 (16× extension): compute the NTK base b' for d=4096. What is the ratio L_target/L_train?
