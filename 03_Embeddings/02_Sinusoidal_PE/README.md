# Understanding Sinusoidal Position Encoding

*The original Transformer's fixed position encoding — why sin/cos and how the math enables relative position awareness*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Position Matters](#11-why-position-matters)
   - [1.2 Pipeline Position](#12-pipeline-position)
2. [The Sinusoidal Formula](#2-the-sinusoidal-formula)
   - [2.1 Full Definition](#21-full-definition)
   - [2.2 Frequency Spectrum](#22-frequency-spectrum)
   - [2.3 Variables Table](#23-variables-table)
3. [Why Sinusoids? Mathematical Justification](#3-why-sinusoids-mathematical-justification)
   - [3.1 Relative Position via Linear Transformation](#31-relative-position-via-linear-transformation)
   - [3.2 Proof of the Rotation Property](#32-proof-of-the-rotation-property)
4. [Numerical Example](#4-numerical-example)
5. [Limitations (Why RoPE Replaced It)](#5-limitations-why-rope-replaced-it)
6. [Common Mistakes](#6-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Position Matters

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Attention is SET-LIKE: it treats input as an unordered set.│
│  Without position info, these give IDENTICAL attention:     │
│                                                             │
│    "The dog bit the man"                                    │
│    "The man bit the dog"  ← completely different meaning!   │
│                                                             │
│  Position encoding adds ORDER information to each token:    │
│    x_t = token_embed(t) + PE(position_t)                    │
│                                                             │
│  Now the model can distinguish position 3 from position 7. │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Imagine reading a book where every page is shuffled. You can read individual sentences but can't follow the plot. Position encoding is like page numbers — it tells the model the ORDER of tokens.

### 1.2 Pipeline Position

```
Token IDs → Embedding lookup → x_t ∈ ℝᵈ
                                  + 
Position → Sinusoidal PE    → PE(t) ∈ ℝᵈ
                                  ↓
                            x_t + PE(t)  → Transformer layers
```

---

## 2. The Sinusoidal Formula

### 2.1 Full Definition

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  PE(pos, 2i)   = sin(pos / 10000^{2i/d})                    │
│  PE(pos, 2i+1) = cos(pos / 10000^{2i/d})                    │
│                                                              │
│  where:                                                      │
│    pos = position in sequence (0, 1, 2, ..., n-1)            │
│    i   = dimension index (0, 1, 2, ..., d/2-1)              │
│    d   = model dimension                                     │
│                                                              │
│  Each position gets a d-dimensional vector of sin/cos values.│
│  Even dimensions use sin, odd dimensions use cos.            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Frequency Spectrum

```
THE KEY: each dimension pair (2i, 2i+1) oscillates at a DIFFERENT frequency.

  Wavelength for dimension pair i:
    λᵢ = 2π × 10000^{2i/d}

  i=0 (first pair):     λ₀ = 2π          (period = 6.28 positions)
  i=d/4 (middle pair):  λ_mid = 2π × 100  (period = 628 positions)
  i=d/2-1 (last pair):  λ_max = 2π × 10000 (period = 62832 positions)

VISUALISATION (d=8, showing 4 frequency pairs):
  Position:  0    1    2    3    4    5    6    7    ...
  dim 0,1:   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (high frequency — local patterns)
  dim 2,3:   ∿∿∿∿∿∿∿∿∿∿          (medium frequency)
  dim 4,5:   ∿∿∿∿∿               (lower frequency)
  dim 6,7:   ∿∿                   (very low — global position)

INTUITION:
  High-freq dims (small i): distinguish NEARBY positions (pos 5 vs pos 6)
  Low-freq dims (large i):  distinguish DISTANT positions (pos 0 vs pos 5000)
  Together: unique "fingerprint" for every position (like binary encoding!)
```

### 2.3 Variables Table

| Symbol | Meaning | Typical value |
|--------|---------|--------------|
| pos | Position in sequence | 0 to n-1 |
| i | Dimension pair index | 0 to d/2-1 |
| d | Model dimension | 512 (original Transformer) |
| 10000 | Base frequency | Fixed constant |
| λᵢ | Wavelength of dim pair i | 2π to 2π×10000 |
| ωᵢ | Angular frequency = 1/10000^{2i/d} | Decreasing with i |

---

## 3. Why Sinusoids? Mathematical Justification

### 3.1 Relative Position via Linear Transformation

```
┌──────────────────────────────────────────────────────────────┐
│  KEY PROPERTY: PE(pos+k) can be expressed as a LINEAR        │
│  TRANSFORMATION of PE(pos), for any fixed offset k.          │
│                                                              │
│  This means: the model can learn to compute RELATIVE position│
│  (distance between tokens) using a simple linear projection. │
│                                                              │
│  FORMALLY: there exists a matrix M_k ∈ ℝ^{d×d} such that:   │
│    PE(pos + k) = M_k × PE(pos)   for ALL pos                │
│                                                              │
│  M_k depends only on k (the offset), NOT on pos.            │
│  → Attention can learn "token 3 positions back" regardless   │
│    of absolute position.                                     │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Proof of the Rotation Property

```
PROOF (for one dimension pair):
  Let ω = 1/10000^{2i/d}  (frequency for this pair)
  
  PE(pos) for dim pair i:
    [sin(ω × pos), cos(ω × pos)]
  
  PE(pos + k) for dim pair i:
    [sin(ω(pos+k)), cos(ω(pos+k))]
  
  Using addition formulas:
    sin(ω(pos+k)) = sin(ωpos)cos(ωk) + cos(ωpos)sin(ωk)
    cos(ω(pos+k)) = cos(ωpos)cos(ωk) - sin(ωpos)sin(ωk)
  
  In matrix form:
    ┌ sin(ω(pos+k)) ┐   ┌ cos(ωk)   sin(ωk) ┐   ┌ sin(ωpos) ┐
    │                │ = │                     │ × │           │
    └ cos(ω(pos+k)) ┘   └ -sin(ωk)  cos(ωk) ┘   └ cos(ωpos) ┘
    
    ↑ PE(pos+k)           ↑ rotation matrix R_k      ↑ PE(pos)
  
  R_k is a 2D ROTATION by angle ωk!
  
  The full d-dimensional transform M_k is a block-diagonal matrix:
    M_k = diag(R_k^{(0)}, R_k^{(1)}, ..., R_k^{(d/2-1)})
  
  Each 2×2 block rotates by a different angle ωᵢ × k.
  This is EXACTLY the idea that RoPE later generalises!  ✓
```

---

## 4. Numerical Example

```
EXAMPLE: d=8, compute PE for positions 0, 1, 2

  Frequencies ωᵢ = 1/10000^{2i/8}:
    ω₀ = 1/10000^{0/8} = 1.0
    ω₁ = 1/10000^{2/8} = 1/10000^{0.25} = 1/10 = 0.1
    ω₂ = 1/10000^{4/8} = 1/10000^{0.5} = 1/100 = 0.01
    ω₃ = 1/10000^{6/8} = 1/10000^{0.75} = 1/1000 = 0.001

  PE(pos=0):
    [sin(0), cos(0), sin(0), cos(0), sin(0), cos(0), sin(0), cos(0)]
  = [0,      1,      0,      1,      0,      1,      0,      1]

  PE(pos=1):
    [sin(1.0), cos(1.0), sin(0.1), cos(0.1), sin(0.01), cos(0.01), sin(0.001), cos(0.001)]
  = [0.841,    0.540,    0.0998,   0.995,    0.01,      1.0,       0.001,      1.0]

  PE(pos=2):
    [sin(2.0), cos(2.0), sin(0.2), cos(0.2), sin(0.02), cos(0.02), sin(0.002), cos(0.002)]
  = [0.909,   -0.416,    0.198,    0.980,    0.02,      1.0,       0.002,      1.0]

OBSERVATION:
  - First pair (ω=1): oscillates rapidly (pos 0→1→2 changes a lot)
  - Last pair (ω=0.001): barely changes (all ≈ [0, 1])
  - Each position gets a UNIQUE pattern (like a binary fingerprint)
```

---

## 5. Limitations (Why RoPE Replaced It)

```
┌─────────────────────────────────────────────────────────────┐
│  LIMITATION 1: ABSOLUTE position is added to content        │
│    x_t = embed(token_t) + PE(t)                             │
│    This mixes content and position → harder for model to    │
│    separate "what" from "where".                            │
│                                                             │
│  LIMITATION 2: Cannot extrapolate beyond training length    │
│    Trained on n=512: PE(513) has sin/cos values never seen! │
│    Model quality degrades at unseen positions.              │
│                                                             │
│  LIMITATION 3: Relative position is only IMPLICIT           │
│    The model CAN learn relative position from PE (via the   │
│    rotation property), but it's indirect — requires the     │
│    attention weights to learn the transformation.           │
│                                                             │
│  ROPE FIXES ALL THREE:                                      │
│    - Encodes RELATIVE position directly in attention scores │
│    - Does NOT add to content (rotates Q/K instead)          │
│    - Better extrapolation properties                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Common Mistakes

```
❌ WRONG: Sinusoidal PE is learned during training
✓ RIGHT:  Sinusoidal PE is FIXED (not learned). The formula gives exact
          values for every position. Learned PE (used in BERT, GPT-2) is
          a separate approach with a learnable matrix P ∈ ℝ^{n_max × d}.

❌ WRONG: The 10000 base is arbitrary and doesn't matter
✓ RIGHT:  10000 determines the frequency RANGE. Larger base → lower
          minimum frequency → can encode longer sequences.
          RoPE uses base=10000 (or 500000 for LLaMA-3) for the same reason.

❌ WRONG: sin and cos carry different information
✓ RIGHT:  sin and cos TOGETHER form a rotation (phase + magnitude).
          One alone is ambiguous: sin(θ) = sin(π-θ). The pair [sin,cos]
          uniquely identifies the angle θ. They are inseparable.

❌ WRONG: Position encoding handles sequences of any length
✓ RIGHT:  While sin/cos are defined for any position, the model has only
          TRAINED on positions 0 to n_max. At positions beyond training,
          attention patterns are unpredictable. Length extrapolation
          requires methods like PI, NTK, or YaRN.
```

---

## 7. Exercises

1. **Uniqueness**: For d=4 and positions 0-7, compute all PE vectors. Verify that no two positions have identical PE vectors. At what maximum position would d=4 start losing uniqueness?

2. **Rotation Verification**: Compute PE(5) and PE(7) for d=4. Verify that PE(7) = M₂ × PE(5) by constructing the 2×2 rotation matrices for k=2.

3. **Frequency Analysis**: For d=512 (original Transformer), compute the wavelength λᵢ for i=0, i=128, and i=255. What is the longest period? Can this encode positions up to 5000?

4. **Dot Product Decay**: Compute PE(0)·PE(k) for k=0,1,2,5,10,50 with d=64. Does the dot product decay with distance? Plot or describe the pattern.
