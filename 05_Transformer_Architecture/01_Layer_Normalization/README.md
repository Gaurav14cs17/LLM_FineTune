# 01 — Layer Normalization & RMSNorm

## 1. The Problem: Activation Scale Drift

```
DEEP NETWORK without normalisation:
  Layer 1: activations ~ N(0, 1)
  Layer 2: activations ~ N(0, 3.2)    (scale drifts up after W multiplication)
  Layer 3: activations ~ N(0, 8.7)
  Layer 10: activations ~ N(0, 1e6)   ← exploding!
  
  Or the reverse:
  Layer 10: activations ~ N(0, 1e-6)  ← vanishing!
  
  Optimizer sees very different gradient magnitudes per layer → very hard to tune.
```

## 2. LayerNorm — Full Derivation

```
Input: x = [x₁, x₂, ..., x_d]

STEP 1: Compute mean
  μ = (1/d) ∑ᵢ xᵢ

STEP 2: Compute variance
  σ² = (1/d) ∑ᵢ (xᵢ − μ)²

STEP 3: Normalise
  x̂ᵢ = (xᵢ − μ) / √(σ² + ε)     ε prevents division by zero

STEP 4: Scale and shift (learned)
  yᵢ = γᵢ × x̂ᵢ + βᵢ

Result: y ~ N(β, γ²) approximately (each dimension independently normalized)

EXAMPLE:
  x = [2, 4, 6, 8]
  μ = 5
  σ² = (9+1+1+9)/4 = 5,   σ = 2.24
  x̂ = [-1.34, -0.45, 0.45, 1.34]
  y = γ × x̂ + β   (γ=1, β=0 at init → y = x̂)

Visualised:
  Before:  2    4    6    8
           │    │    │    │
           ▼    ▼    ▼    ▼
  After: -1.34 -0.45 0.45 1.34  ← zero mean, unit variance
```

## 3. RMSNorm — Simplified Version

```
RMSNorm omits the mean subtraction step:

  RMS(x) = √( (1/d) ∑ᵢ xᵢ² )      ← root mean square

  y = γ ⊙ (x / RMS(x))

COMPARISON:
  LayerNorm:  subtract mean → divide by std → scale (γ) → shift (β)
  RMSNorm:    divide by RMS               → scale (γ)
                                           NO shift parameter!

WHY REMOVE MEAN SUBTRACTION?
  ┌─────────────────────────────────────────────────────────────┐
  │  1. The β (shift) parameter is redundant with the bias      │
  │     in the next linear layer. Removing it doesn't hurt.    │
  │                                                            │
  │  2. Mean subtraction requires an extra pass over d elements│
  │     → ~15% speed improvement with RMSNorm                  │
  │                                                            │
  │  3. Empirically: same quality on LM benchmarks             │
  └─────────────────────────────────────────────────────────────┘

Parameters:  LayerNorm = 2d (γ and β)   RMSNorm = d (γ only)
```

## 4. Pre-Norm vs Post-Norm

```
ORIGINAL TRANSFORMER (Post-Norm):
  Input x
     │
     ├──────────────────────┐
     │                      │
     ▼                      │
  Attention(x)              │
     │                      │
     ▼                      │
  + ←───────────────────────┘  (residual add)
     │
     ▼
  LayerNorm                 ← norm AFTER residual
     │
  output

MODERN LLMs (Pre-Norm, e.g. LLaMA):
  Input x
     │
     ├──────────────────────┐
     │                      │
     ▼                      │
  RMSNorm(x)                │
     │                      │
     ▼                      │
  Attention(...)            │
     │                      │
     ▼                      │
  + ←───────────────────────┘  (residual add)
     │
  output

GRADIENT COMPARISON:
  Post-norm: ∂L/∂x flows through LayerNorm → can have scaling issues
  Pre-norm:  ∂L/∂x = ∂L/∂(x + F(RMSNorm(x)))/∂x = I + Jacobian
                     ↑ identity term guarantees non-zero gradient!
```

## 5. Visualising Normalisation Effect

```
BEFORE LayerNorm (after ATTN layer, unnormalised):
  Token 0:  [12.3, -45.2,  88.1, -3.4, ...]   (large range!)
  Token 1:  [ 0.1,   2.3,  -1.2,  0.8, ...]   (small range)
  Token 2:  [-99.0, 120.0, -55.0, 33.0, ...]  (very large range)

AFTER LayerNorm:
  Token 0:  [-0.4,  -1.8,   2.1,  -0.3, ...]  ← standardised
  Token 1:  [-0.2,   1.8,  -0.9,   0.5, ...]  ← standardised
  Token 2:  [-1.1,   1.4,  -0.6,   0.3, ...]  ← standardised
  
  Now all tokens have similar scale → stable training!
```

---

## Exercises

1. For x = [1, 2, 3, 4], compute LayerNorm(x) with γ=1, β=0, ε=0.
2. Repeat with RMSNorm. Compare to LayerNorm.
3. Show that if all xᵢ are shifted by a constant c (x → x + c·1), RMSNorm changes but LayerNorm does not. Why does this matter?
