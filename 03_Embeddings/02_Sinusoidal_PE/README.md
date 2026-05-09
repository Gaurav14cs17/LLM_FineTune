# 02 — Sinusoidal Positional Encoding

## 1. Why Position Encoding is Necessary

```
TRANSFORMER ATTENTION IS PERMUTATION EQUIVARIANT:
  For any permutation π of token positions:
    Attention(X_π) = Attention(X)_π   (attention output for permuted input = permuted output)
  
  PROOF:
    Score matrix S = QKᵀ/√d_k.
    If we permute X → X_π:
      Q_π = X_π · W_Q,  K_π = X_π · W_K
      S_π = Q_π · K_πᵀ = (X_π W_Q)(X_π W_K)ᵀ
    This equals a permuted version of S.
    After softmax and V multiplication: output is also permuted. QED.
  
  CONSEQUENCE: "The cat sat" and "sat cat The" produce the SAME output vectors
               (just in different order). The model cannot distinguish word ORDER!

POSITION ENCODING fixes this by adding position-specific information to each token:
  x̃_t = E[w_t] + PE(t)
  Now each position has a unique input → equivariance is broken.
```

## 2. The Sinusoidal Formula — Complete Analysis

```
VASWANI ET AL. (2017) proposed:
  PE(t, 2i)   = sin(t / 10000^{2i/d})
  PE(t, 2i+1) = cos(t / 10000^{2i/d})
  
  for i = 0, 1, ..., d/2 − 1  (dimension pair index)

FREQUENCIES:
  ω_i = 1 / 10000^{2i/d}  (angular frequency for pair i)
  
  i = 0:   ω₀ = 1/10000⁰ = 1.0            (fastest: completes full cycle in 2π ≈ 6.3 positions)
  i = 1:   ω₁ = 1/10000^{2/d}              (slower)
  ...
  i = d/2: ω_{d/2-1} = 1/10000^{(d-2)/d}  (slowest: cycle length ≈ 2π×10000 = 62831 positions!)
  
  GEOMETRIC RANGE:
    Maximum cycle length (slowest freq): 2π / ω_{min} = 2π × 10000^{(d-2)/d}
    For d=512: = 2π × 10000^{510/512} ≈ 2π × 9949 ≈ 62500 positions
    For d=4096: ≈ 2π × 10000^{4094/4096} ≈ 62830 positions
    
    → Can uniquely represent positions up to ~62K!

MATRIX FORM:
  PE(t) ∈ ℝ^d is the full positional encoding vector for position t.
  
  Entries:
    PE(t)[2i]   = sin(ω_i × t)
    PE(t)[2i+1] = cos(ω_i × t)

  For t=0: PE(0) = [sin(0),cos(0), sin(0),cos(0), ...] = [0,1, 0,1, ...]
  For t=1: PE(1) = [sin(1),cos(1), sin(ω₁),cos(ω₁), ...]
```

## 3. The Relative Position Property — Proof

```
CLAIM: PE(t+k) can be written as a LINEAR function of PE(t).

PROOF (in the 2D subspace for each pair i):
  PE(t)_pair_i = [sin(ω_i t), cos(ω_i t)]
  PE(t+k)_pair_i = [sin(ω_i(t+k)), cos(ω_i(t+k))]
                 = [sin(ω_i t + ω_i k), cos(ω_i t + ω_i k)]
  
  Using angle addition formulas:
    sin(α+β) = sin(α)cos(β) + cos(α)sin(β)
    cos(α+β) = cos(α)cos(β) − sin(α)sin(β)
  
  So:
    sin(ω_i(t+k)) = sin(ω_i t) cos(ω_i k) + cos(ω_i t) sin(ω_i k)
    cos(ω_i(t+k)) = cos(ω_i t) cos(ω_i k) − sin(ω_i t) sin(ω_i k)
  
  MATRIX FORM:
  ┌ sin(ω_i(t+k)) ┐   ┌ cos(ω_i k)   sin(ω_i k) ┐ ┌ sin(ω_i t) ┐
  │               │ = │                           │ │            │
  └ cos(ω_i(t+k)) ┘   └ −sin(ω_i k)  cos(ω_i k) ┘ └ cos(ω_i t) ┘
  
                     = R(ω_i k) × PE(t)_pair_i
  
  where R(θ) is a 2×2 rotation matrix!

CONSEQUENCE for attention:
  PE(t+k) = M_k × PE(t)   where M_k is block-diagonal with 2×2 rotations.
  
  The attention score between PE(t_q) and PE(t_k):
    PE(t_q)ᵀ · PE(t_k) = [M_{t_q} PE(0)]ᵀ [M_{t_k} PE(0)]
                        = PE(0)ᵀ M_{t_q}ᵀ M_{t_k} PE(0)
                        = PE(0)ᵀ M_{t_k-t_q} PE(0)   ← depends only on t_k − t_q!
  
  → Sinusoidal PE encodes RELATIVE positions in the dot product! ■
  (This is the same property that RoPE achieves, but in the content-free PE,
   not applied to Q,K directly as in RoPE.)
```

## 4. Norm Analysis

```
‖PE(t)‖² = ∑_{i=0}^{d/2-1} [sin²(ω_i t) + cos²(ω_i t)]
          = ∑_{i=0}^{d/2-1} 1    [by Pythagorean identity sin²+cos²=1]
          = d/2

‖PE(t)‖ = √(d/2)

For d=4096: ‖PE(t)‖ = √2048 ≈ 45.3

IMPACT ON EMBEDDING SCALE:
  Embedding E[t] ~ N(0, 1/d) → ‖E[t]‖ ≈ 1
  PE: ‖PE(t)‖ ≈ √(d/2) ≈ 45
  
  WITHOUT scaling: PE would DOMINATE the token identity!
  
  In original Transformer: scale embedding by √d:
    ‖E[t] × √d‖ ≈ 1 × √d = 64  ≈ ‖PE(t)‖  ← balanced ✓

MODERN LLMs:
  Use RoPE instead of sinusoidal PE (applied to Q,K, not added to input).
  RMSNorm at each layer normalises activations regardless.
  → No explicit embedding scaling needed.
```

## 5. Learned vs Sinusoidal PE

```
SINUSOIDAL (Vaswani et al.):
  Fixed formula, no parameters.
  Advantages:
    + Zero extra parameters
    + Handles sequences LONGER than training length (extrapolation)
      (e.g., trained on L=512, can encode L=1000 positions)
    + Smooth, structured representation
  Disadvantages:
    - Sub-optimal: not learned from data
    - Extrapolation quality degrades for positions >> L_train

LEARNED PE (BERT, GPT-2):
  Each position t gets a trainable vector pₜ ∈ ℝ^d.
  Embedding table P ∈ ℝ^{L_max × d} (learned).
  
  Advantages:
    + Can adapt PE to data distribution
    + Empirically slightly better within L_max range
  Disadvantages:
    - Extra L_max × d parameters (for L=2048, d=1024: 2M extra params)
    - CANNOT generalise beyond L_max! (no position vector for t > L_max)
    - Positions near L_max are rarely trained → poor representation

ROTARY PE (ROPE) — used in LLaMA:
  Applied to Q,K before dot-product (not added to input).
  Properties:
    + Handles relative positions naturally (same as sinusoidal advantage)
    + Can be extended beyond L_train via scaling (NTK, YaRN methods)
    + No extra parameters
    + Better empirical performance than both sinusoidal and learned PE

COMPARISON TABLE:
  Method     │ Params │ Extrapolates │ Relative Pos │ Performance
  ───────────┼────────┼──────────────┼──────────────┼─────────────
  Sinusoidal │   0    │      ✓       │     partly   │  baseline
  Learned    │ L×d    │      ✗       │      ✗       │  +0.5%
  ALiBi      │   0    │      ✓       │      ✓       │  +1-2%
  RoPE       │   0    │    ✓(w/ext)  │      ✓       │  +1-3%
```

---

## Exercises

1. For d=8, compute PE(t=3) fully — all 8 entries — using the sinusoidal formula with base 10000.
2. Prove sin²(θ) + cos²(θ) = 1 and use it to show ‖PE(t)‖ = √(d/2) regardless of t. What does this imply about the position encoding magnitude?
3. Show that the dot product PE(t)·PE(t+k) can be written as a function of k only (not t). This is the relative-position property. Evaluate PE(0)·PE(3) for d=4, base=10000.
