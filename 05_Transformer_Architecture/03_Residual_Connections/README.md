# 03 — Residual Connections

## 1. The Vanishing Gradient Problem — Formal Analysis

```
DEEP NETWORK (no residuals), L layers:
  y = f_L ∘ f_{L-1} ∘ ... ∘ f_1(x)

GRADIENT via chain rule:
  ∂L/∂x = (∂f_L/∂f_{L-1}) × (∂f_{L-1}/∂f_{L-2}) × ... × (∂f_1/∂x)
         = ∏_{l=1}^{L} J_l   where J_l = ∂f_l/∂f_{l-1} is the Jacobian

SPECTRAL NORM ANALYSIS:
  ‖∂L/∂x‖ ≤ ‖∂L/∂y‖ × ∏_{l=1}^{L} ‖J_l‖_2

  If ‖J_l‖_2 = ρ < 1 for each layer:
    ‖∂L/∂x‖ ≤ ‖∂L/∂y‖ × ρ^L

  For ρ = 0.9, L = 32:   0.9^{32} = 0.034   → gradient shrinks to 3.4%
  For ρ = 0.9, L = 96:   0.9^{96} ≈ 0.0003  → gradient shrinks to 0.03%
  
  ┌──────────────────────────────────────────────────────┐
  │  Gradient magnitude flowing from layer 32 to layer 1 │
  │                                                      │
  │  L=1 ████████████████████████████████ 100%           │
  │  L=4 ███████████████████████ 66%                     │
  │  L=8 ████████████████ 43%                            │
  │  L=16 ██████████ 19%                                 │
  │  L=32 ████ 3.4%  ← deep layers barely learn!        │
  │  L=64 █ 0.1%                                         │
  └──────────────────────────────────────────────────────┘
```

## 2. Residual Connection — Full Derivation

```
RESIDUAL BLOCK:
  Output = x + F(x)       ("skip" or "shortcut" connection)

GRADIENT COMPUTATION:
  ∂(x + F(x))/∂x = I + ∂F/∂x

  The identity matrix I acts as a "gradient highway":
  even if ∂F/∂x ≈ 0 (layer learns identity), gradient still flows via I!

FULL GRADIENT THROUGH L RESIDUAL BLOCKS:
  y_L = y_0 + ∑_{l=1}^{L} F_l(y_{l-1})   ← telescope the residuals
  
  ∂L/∂y_0 = ∂L/∂y_L × ∂y_L/∂y_0
           = ∂L/∂y_L × (I + ∑_l ∂F_l/∂y_0)
                       ↑ identity: direct gradient path from output to input!

PROOF of telescoped form:
  y_1 = y_0 + F_1(y_0)
  y_2 = y_1 + F_2(y_1) = y_0 + F_1(y_0) + F_2(y_1)
  ...
  y_L = y_0 + ∑_{l=1}^{L} F_l(y_{l-1})   ■

KEY INSIGHT: The gradient ∂L/∂y_0 always contains the term ∂L/∂y_L (identity path),
regardless of how many layers intervene. Gradient CANNOT vanish through the skip connections.
```

## 3. "Learning Residuals" — Why Residuals Learn Easier

```
WITHOUT residuals, the l-th layer must learn the FULL mapping:
  y_l = H_l(y_{l-1})   (arbitrary transformation)

WITH residuals, the l-th layer only needs to learn the RESIDUAL:
  y_l = y_{l-1} + F_l(y_{l-1})
  F_l(x) = H_l(x) − x   ← the "residual" relative to identity

ADVANTAGE at initialisation:
  If optimal transformation is close to identity (common in fine-tuning!):
    H_l(x) ≈ x  →  F_l(x) ≈ 0
    
  Learning F_l → 0 is easy (set weights ≈ 0).
  Learning H_l → identity is hard (identity matrix is a specific solution to find).

LEARNING DYNAMICS:
  Early training: F_l ≈ 0 (weights near zero), network acts as identity.
    ‖output − input‖ ≈ 0 for all layers
    Gradients flow easily → stable start

  Mid training: F_l learns increasingly complex residuals.
    Each layer adds small but meaningful changes.

  Late training / fine-tuning (LoRA insight):
    Fine-tuning updates ΔW ≈ small residual on W_pretrained
    LoRA explicitly models this: W = W₀ + BA (residual structure!)
```

## 4. Norm Preservation Analysis

```
CONCERN: do residuals cause exploding activations?

ANALYSIS for a single residual block y = x + F(x):

  ‖y‖² = ‖x + F(x)‖²
        = ‖x‖² + 2⟨x, F(x)⟩ + ‖F(x)‖²

  If F is initialised with small weights: ‖F(x)‖ ≈ ε (small)
  And ⟨x, F(x)⟩ ≈ 0 (random initialisation, uncorrelated)

  ‖y‖² ≈ ‖x‖² + ε²   ← only slightly grows

  AFTER L layers: ‖y_L‖² ≈ ‖y_0‖² + L×ε²
  
  For L=32 and ε=0.1:  ‖y_L‖² ≈ ‖y_0‖² + 0.32  (negligible growth!)

  This is MUCH better than without residuals where ‖y_L‖ can grow as ρ^L.

IN PRACTICE: LayerNorm / RMSNorm before each block controls the scale:
  y = x + F(Norm(x))
  ‖Norm(x)‖ = √d  always  → consistent scale regardless of depth!
```

## 5. Ensemble Interpretation

```
MATHEMATICAL CLAIM (Veit et al. 2016):
  A depth-L residual network is an implicit ensemble of 2^L sub-networks!

DERIVATION:
  y = y₀ + F₁(y₀) + F₂(y₀+F₁(y₀)) + ...

  At each block, the signal can "use" or "skip" that block.
  If we model each Fₗ as either present (weight 1) or absent (weight 0):
    There are 2^L possible subsets of blocks
    Each subset defines a distinct sub-network
    The full network = weighted average of all sub-networks

  For L=32: 2^32 ≈ 4 billion sub-networks!
  
  ┌─────────────────────────────────────────────────────────┐
  │  Sub-network 1:  uses blocks {1, 3, 5, ...}             │
  │  Sub-network 2:  uses blocks {1, 2, 4, ...}             │
  │  ...                                                    │
  │  Sub-network 2^L: uses all blocks                       │
  │                                                         │
  │  Full network output ≈ average over all sub-networks    │
  │  → rich ensemble behaviour, robust predictions          │
  └─────────────────────────────────────────────────────────┘

CONSEQUENCE: Removing individual layers at inference has little effect.
Shown empirically: removing one layer of a 32-layer Transformer drops
accuracy by <2%, because the remaining 31 layers compensate.
```

---

## Exercises

1. Show that for a 2-layer network without residuals, the gradient is a product of two Jacobians. Add a residual to layer 1 and show how the gradient calculation changes.
2. In fine-tuning with LoRA, the full weight is W = W₀ + AB. Identify the "residual" here and explain why it's easier to learn than relearning W from scratch.
3. If each residual block adds ‖F_l(x)‖ = 0.05×‖x‖ in magnitude, compute how much the activation norm grows after L=32 layers. Is this concerning?
