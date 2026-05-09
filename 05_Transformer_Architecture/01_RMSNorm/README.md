# Understanding RMSNorm

*A comprehensive guide to normalisation in modern LLMs*

---

**RMSNorm** (Root Mean Square Normalisation) is the normalisation layer used in LLaMA, Mistral, Falcon, and most modern LLMs. It is a simpler and faster alternative to LayerNorm that achieves comparable or better results.

This guide explains the mathematical differences, the gradient flow analysis, and why modern LLMs prefer RMSNorm over the original LayerNorm.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Do We Need Normalisation?](#11-why-do-we-need-normalisation)
   - [1.2 LayerNorm vs RMSNorm at a Glance](#12-layernorm-vs-rmsnorm-at-a-glance)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [LayerNorm: The Baseline](#2-layernorm-the-baseline)
   - [2.1 Full Formula](#21-full-formula)
3. [RMSNorm: The Simplification](#3-rmsnorm-the-simplification)
   - [3.1 Full Formula](#31-full-formula)
   - [3.2 Why No Mean Centering?](#32-why-no-mean-centering)
   - [3.3 Scale Invariance Proof](#33-scale-invariance-proof)
4. [Gradient Flow Analysis](#4-gradient-flow-analysis)
   - [4.1 Jacobian of RMSNorm](#41-jacobian-of-rmsnorm)
5. [Pre-Norm vs Post-Norm](#5-pre-norm-vs-post-norm)
   - [5.1 Training Stability Analysis](#51-training-stability-analysis)
6. [Summary](#6-summary)
   - [6.1 All Formulas Quick Reference](#61-all-formulas-quick-reference)
   - [6.2 Common Mistakes](#62-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Do We Need Normalisation?

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Vanishing/Exploding Activations in Deep Networks  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Without normalisation, activation magnitude in layer l:    │
│    ‖h_l‖ ≈ ρˡ × ‖h_0‖  where ρ depends on weight init     │
│                                                             │
│  ρ < 1 → ‖h_l‖ → 0  (vanishing activations)               │
│  ρ > 1 → ‖h_l‖ → ∞  (exploding activations)               │
│                                                             │
│  Example (ρ = 0.9, 32-layer network):                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Layer 1:  ‖h‖ = 1.000  ████████████████████           │ │
│  │ Layer 8:  ‖h‖ = 0.430  █████████                      │ │
│  │ Layer 16: ‖h‖ = 0.185  ████                           │ │
│  │ Layer 32: ‖h‖ = 0.034  █                              │ │
│  └────────────────────────────────────────────────────────┘ │
│  → Early layers get almost NO gradient signal!             │
│                                                             │
│  NORMALISATION FIX: constrain ‖h_l‖ = constant at each l  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 LayerNorm vs RMSNorm at a Glance

| Feature | LayerNorm (Ba et al. 2016) | RMSNorm (Zhang & Sennrich 2019) |
|---------|--------------------------|--------------------------------|
| Centering | Subtracts mean μ | ❌ No mean subtraction |
| Scaling | Divides by std σ | Divides by RMS |
| Learnable params | γ (scale) + β (shift) | γ (scale) only |
| Parameters per layer | 2d | d (50% fewer!) |
| Speed | Baseline | ~20-30% faster |
| Used in | BERT, GPT-2, T5 | LLaMA, Mistral, Falcon |

### 1.3 Pipeline Summary

```
INPUT: x ∈ ℝᵈ  (one token's d-dimensional representation)

LAYERNORM:
  μ = mean(x)  →  σ² = var(x)  →  x̂ = (x−μ)/√(σ²+ε)  →  γ⊙x̂ + β

RMSNORM (simpler):
  RMS(x) = √(mean(x²))  →  x̂ = x/RMS(x)  →  γ⊙x̂

OUTPUT: normalised x̂ with learnable scale γ
```

---

## 2. LayerNorm: The Baseline

### 2.1 Full Formula

```
STEP 1: Compute mean
  μ = (1/d) × Σᵢ xᵢ

STEP 2: Compute variance
  σ² = (1/d) × Σᵢ (xᵢ − μ)²

STEP 3: Normalise
  x̂ᵢ = (xᵢ − μ) / √(σ² + ε)       ε = 1e-5 (stability)

STEP 4: Rescale and shift
  LN(x)ᵢ = γᵢ × x̂ᵢ + βᵢ
  γ, β ∈ ℝᵈ are learnable parameters

EXAMPLE (d=4, x=[1, 3, 5, 7]):
  μ = (1+3+5+7)/4 = 4.0
  σ² = [(1-4)²+(3-4)²+(5-4)²+(7-4)²]/4 = [9+1+1+9]/4 = 5.0
  x̂ = [(-3)/√5, (-1)/√5, (1)/√5, (3)/√5]
     = [-1.342, -0.447, 0.447, 1.342]
  LN(x) = γ⊙x̂ + β   (with γ=[1,1,1,1], β=[0,0,0,0]: same as x̂)
```

---

## 3. RMSNorm: The Simplification

### 3.1 Full Formula

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  RMS(x) = √( (1/d) × Σᵢ xᵢ² )   =  √( ‖x‖² / d )          │
│                                                              │
│  RMSNorm(x)ᵢ = γᵢ × xᵢ / RMS(x)                            │
│                                                              │
│  → No mean subtraction                                       │
│  → No shift parameter β                                      │
│  → Only scale parameter γ ∈ ℝᵈ                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘

EXAMPLE (d=4, x=[1, 3, 5, 7]):
  RMS(x) = √( (1+9+25+49)/4 ) = √(84/4) = √21 ≈ 4.583
  RMSNorm(x) = [1/4.583, 3/4.583, 5/4.583, 7/4.583]
              = [0.218, 0.655, 1.091, 1.527]
              (with γ=[1,1,1,1])

COMPARISON to LayerNorm:
  LayerNorm output: [-1.342, -0.447, 0.447, 1.342]
  RMSNorm output:  [ 0.218,  0.655, 1.091, 1.527]
  
  LayerNorm: CENTRED around 0 (mean subtracted)
  RMSNorm:   NOT centred, just scaled to unit RMS
```

### 3.2 Why No Mean Centering?

```
┌─────────────────────────────────────────────────────────────┐
│  ARGUMENT: Re-centering is REDUNDANT when biases exist      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LayerNorm subtracts mean: LN(x) = γ(x−μ)/σ + β           │
│                                                             │
│  The shift β can ABSORB any constant offset.               │
│  So subtracting μ then adding β is equivalent to           │
│  NOT subtracting μ and learning a different β.             │
│                                                             │
│  In the Transformer, every linear layer has a bias:        │
│  y = x×W + b                                               │
│                                                             │
│  The bias b can compensate for any fixed mean shift.       │
│  → Centering is redundant! RMSNorm skips it safely.        │
│                                                             │
│  WHAT IS NOT REDUNDANT: the scale normalisation.           │
│  Without dividing by RMS/std:                              │
│    activations grow across layers → exploding values       │
│  Biases CANNOT compensate for magnitude growth.            │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Scale Invariance Proof

```
CLAIM: RMSNorm(c×x) = RMSNorm(x)  for any scalar c > 0.

PROOF:
  RMS(c×x) = √( (1/d) Σᵢ (c×xᵢ)² )
            = √( c² × (1/d) Σᵢ xᵢ² )
            = c × RMS(x)

  RMSNorm(c×x)ᵢ = γᵢ × (c×xᵢ) / RMS(c×x)
                = γᵢ × (c×xᵢ) / (c×RMS(x))
                = γᵢ × xᵢ / RMS(x)
                = RMSNorm(x)ᵢ   ■

INTERPRETATION:
  If you scale ALL features in x by the same factor c,
  RMSNorm gives EXACTLY the same output.
  
  This means the output depends only on the DIRECTION of x in ℝᵈ,
  not its magnitude. This is the desired normalisation behaviour.
```

---

## 4. Gradient Flow Analysis

### 4.1 Jacobian of RMSNorm

```
For RMSNorm(x)ᵢ = γᵢ × xᵢ / r   where r = RMS(x):

∂RMSNorm(x)ᵢ / ∂xⱼ = ?

CASE j = i:
  ∂(γᵢ xᵢ/r) / ∂xᵢ = γᵢ/r − γᵢ xᵢ (∂r/∂xᵢ) / r²

  ∂r/∂xᵢ = xᵢ/(d × r)   [derivative of √(Σxⱼ²/d) w.r.t. xᵢ]

  = γᵢ/r × (1 − xᵢ²/(d×r²))
  = (γᵢ/r) × (1 − (xᵢ/r)²/d)

CASE j ≠ i:
  ∂(γᵢ xᵢ/r) / ∂xⱼ = −γᵢ xᵢ × xⱼ / (d × r³)

JACOBIAN STRUCTURE:
  J[i,j] = (γᵢ/r) × [δᵢⱼ − (xᵢ/r)(xⱼ/r)/d]
          = (γᵢ/r) × [I − x̂x̂ᵀ/d]ᵢⱼ

  This is a PROJECTION matrix: it projects out the radial direction (scaling).
  
  Gradients TANGENT to the unit sphere pass through unchanged.
  Gradients ALONG the radial direction (pure scaling) → 0.
  → RMSNorm prevents gradients from "flowing via scale" which is not meaningful.
```

---

## 5. Pre-Norm vs Post-Norm

### 5.1 Training Stability Analysis

```
┌─────────────────────────────────────────────────────────────┐
│  ORIGINAL TRANSFORMER (Post-Norm):                          │
│    y = Norm(x + Attention(x))                               │
│    y = Norm(y + FFN(y))                                     │
│                                                             │
│  GRADIENT PATH (Post-Norm):                                 │
│    x ──→ Attention ──→ + ──→ Norm ──→ output                │
│    ↑                    ↑      ↑                            │
│    gradient must PASS THROUGH Norm at every layer           │
│    → Norm Jacobian can slow down gradient flow              │
│    → Training with 12+ layers often diverges without warmup │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  MODERN LLMs (Pre-Norm):  Used in LLaMA, Mistral, GPT-3     │
│    y = x + Attention(Norm(x))                               │
│    y = y + FFN(Norm(y))                                     │
│                                                             │
│  GRADIENT PATH (Pre-Norm):                                  │
│    x ──────────────────────── + ──→ output                  │
│    └──→ Norm ──→ Attention ──┘↑                             │
│                                                             │
│    The IDENTITY PATH (x → output) is unobstructed!         │
│                                                             │
│  PROOF: ∂L/∂x = ∂L/∂y × (∂y/∂x)                          │
│    y = x + Attention(Norm(x))                              │
│    ∂y/∂x = I + ∂Attention(Norm(x))/∂x                     │
│                                                             │
│    The identity matrix I provides a DIRECT gradient path.  │
│    Even if Attention contributes 0 gradient:               │
│    ∂L/∂x = ∂L/∂y × I = ∂L/∂y  ← always flows! ✓          │
└─────────────────────────────────────────────────────────────┘

TRAINING STABILITY COMPARISON:
  Post-Norm: training can diverge without careful LR warmup
  Pre-Norm:  more stable, can train with less warmup
  
  Both eventually converge to similar quality,
  but Pre-Norm is safer for deep models (32+ layers).
```

---

## 6. Summary

### 6.1 All Formulas Quick Reference

**LayerNorm:**

```
μ = (1/d) Σᵢ xᵢ
σ² = (1/d) Σᵢ (xᵢ−μ)²
LN(x)ᵢ = γᵢ × (xᵢ−μ)/√(σ²+ε)  +  βᵢ
```

**RMSNorm:**

```
RMS(x) = √( (1/d) Σᵢ xᵢ² )
RMSNorm(x)ᵢ = γᵢ × xᵢ / RMS(x)
```

**Scale invariance:**

```
RMSNorm(c×x) = RMSNorm(x)   for any c > 0
```

**Pre-Norm transformer block:**

```
y = x + Attention(RMSNorm(x))
y = y + FFN(RMSNorm(y))
```

| Property | LayerNorm | RMSNorm |
|----------|-----------|---------|
| ‖output‖ after norm | ‖γ‖ (controlled) | ‖γ‖ / √d × ‖x‖/RMS ≈ ‖γ‖ |
| Invariant to | Shift AND scale of x | Scale of x only |
| Parameters per layer | 2d (γ + β) | d (γ only) |
| Speed | 1× | ~1.2-1.3× faster |

### 6.2 Common Mistakes

```
❌ WRONG: RMSNorm(x) has mean 0 after normalisation
✓ RIGHT:  Only LayerNorm guarantees mean 0. RMSNorm normalises the
          magnitude (L2 norm) but does NOT centre. The mean of
          RMSNorm(x) is generally non-zero.

❌ WRONG: Post-Norm is strictly inferior to Pre-Norm
✓ RIGHT:  Post-Norm (BERT, original Transformer) works fine for shallow
          models. Pre-Norm is preferred for very deep models (32+ layers).
          With proper warmup, both can train well.

❌ WRONG: ε (epsilon) in LayerNorm is needed for RMSNorm too
✓ RIGHT:  RMSNorm doesn't compute σ², so there's no variance → 0 issue.
          However, RMS(x) = 0 is possible (all-zero input) so ε is still
          sometimes added: RMS(x) = √(mean(x²) + ε) for safety.

❌ WRONG: The γ parameter in RMSNorm is initialised to 0
✓ RIGHT:  γ is initialised to 1 (ones), so RMSNorm(x) ≈ x/RMS(x) at init.
          If γ=0, the normalised output is always 0 → dead layer!
```

---

## 7. Exercises

1. **RMSNorm Computation**: For x = [2, 4, 6, 8] with γ = [1, 1, 1, 1], compute RMS(x) and RMSNorm(x) step by step. Compare with LayerNorm output.

2. **Scale Invariance**: Verify RMSNorm(3x) = RMSNorm(x) for x = [1, 2, 3]. Show that multiplying all elements by 3 doesn't change the normalised output.

3. **Parameter Savings**: LLaMA-3 8B has 32 layers, d=4096. How many parameters are saved by using RMSNorm instead of LayerNorm across all 32 normalisation layers? (Hint: each layer uses 2 norm layers — one before attention, one before FFN.)

4. **Pre-Norm Gradient**: Show algebraically that in Pre-Norm (y = x + f(Norm(x))), the gradient ∂L/∂x always contains ∂L/∂y as a term, regardless of the function f. Why is this "gradient highway" beneficial for training?
