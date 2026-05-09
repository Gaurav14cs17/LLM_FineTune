# 02 — AdamW Optimizer

## 1. Why Not Plain SGD?

```
SGD: θ ← θ − η · ∇L

PROBLEM 1 — Different parameter scales:
  Embedding rows for rare tokens: gradient magnitude ~0.001 (rare, few updates)
  Common token embeddings:        gradient magnitude ~10.0  (frequent)
  SGD uses same η for both → either rare tokens don't update, or common ones explode

PROBLEM 2 — Oscillations in ravines:
  Loss landscape looks like a narrow canyon:
       ↑ y
       │  ╲  ╱  ╲  ╱    ← SGD bounces back and forth across the canyon
       │   ╲╱    ╲╱     (huge steps perpendicular to optimum)
  ─────┼───────────────── x
       │  (slow progress along the canyon floor)
  
ADAM solves both by tracking per-parameter adaptive learning rates.
```

## 2. Adam — Full Algorithm with Motivation

```
State per parameter: mₜ (1st moment), vₜ (2nd moment)

INITIALISE: m₀ = 0,  v₀ = 0,  t = 0

EACH STEP:
  t ← t + 1
  gₜ = ∇_θ L(θₜ₋₁)          COMPUTE gradient

  mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ  RUNNING AVERAGE of gradient (momentum)
       ↑ forget old           ↑ include new
       
  vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ² RUNNING AVERAGE of squared gradient
       (tracks per-parameter gradient variance)

  BIAS CORRECTION (counters initialisation from 0):
  m̂ₜ = mₜ / (1 − β₁ᵗ)       At t=1: m̂₁ = m₁/(1-0.9) = 10×m₁ ← boosts early signal
  v̂ₜ = vₜ / (1 − β₂ᵗ)       Fades as t grows (β₁ᵗ → 0)

  UPDATE:
  θₜ = θₜ₋₁ − η · m̂ₜ / (√v̂ₜ + ε)

  INTERPRETATION:
    m̂ₜ / √v̂ₜ  ≈  (avg gradient direction) / (gradient noise)
                 = signal-to-noise ratio estimate
    Large noise (oscillating grad) → small effective step
    Consistent direction             → large effective step
```

## 3. Adam vs AdamW — Weight Decay

```
STANDARD ADAM with L2 regularisation (WRONG):
  Add λ/2 ‖θ‖² to loss
  Gradient becomes: g = ∇loss + λθ
  Adam update: θ ← θ − η · (m̂/√v̂)  where m̂ includes λθ
  
  PROBLEM: weight decay term is divided by √v̂ (adaptive scaling)
  For params with high gradient noise (large v̂):
    effective weight decay = λ × (η/√v̂) → TINY (under-regularised!)

  ┌──────────────────────────────────────────────────────┐
  │  Parameter A: consistent grad, small v̂              │
  │    Weight decay: λη/√v̂ = large  ✓                   │
  │  Parameter B: noisy grad, large v̂                   │
  │    Weight decay: λη/√v̂ = tiny   ✗ (not regularised!) │
  └──────────────────────────────────────────────────────┘

ADAMW — DECOUPLED weight decay (CORRECT):
  θₜ = θₜ₋₁  −  η · m̂ₜ/(√v̂ₜ + ε)   −  η·λ·θₜ₋₁
                 ─────────────────────   ──────────────
                 adaptive gradient step  CONSTANT weight decay (same for all params)
```

## 4. Gradient Clipping

```
BEFORE update, clip gradients by global norm:

  COMPUTE global gradient norm:
  g_norm = √( ‖g_{layer1}‖² + ‖g_{layer2}‖² + ... + ‖g_{layerN}‖² )
         = √( ∑ all_params (∂L/∂θᵢ)² )

  IF g_norm > clip_max (= 1.0):
    scale = clip_max / g_norm
    gᵢ ← gᵢ × scale   for ALL parameters i

VISUAL:
  gradient vector g in parameter space:
                            Before clip
  ┌──────────────────────────────────────────────────────┐
  │   g is a vector in ℝ^{N_params}                      │
  │   ‖g‖ = 8.3   (too large, > clip_max=1.0)            │
  │                                                      │
  │       g →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→          │
  └──────────────────────────────────────────────────────┘
  After clip (scale = 1.0/8.3 = 0.12):
  ┌──────────────────────────────────────────────────────┐
  │   g_clipped →→                                       │
  │   ‖g_clipped‖ = 1.0 ✓   (same DIRECTION, smaller)   │
  └──────────────────────────────────────────────────────┘

WHY: Bad data batches can cause huge gradients → training instability.
Clipping caps the worst case while preserving gradient direction.
```

## 5. AdamW Training Trajectory

```
Loss
  │\
  │ \   Warmup phase:  LR ramps up  → aggressive early learning
  │  \
  │   ─\  Stable phase: LR at max → fast convergence
  │     \
  │      ─\  Decay phase: LR falls → fine precision tuning
  │         ──────────────────────────────────────────────
  └────────────────────────────────────────── steps
       warmup  ─────────────── cosine decay ────────────

AdamW adaptive steps (per param perspective):
  param A (embedding, high grad):  large steps early, settles to final value
  param B (LayerNorm, small grad): small steps throughout, high precision
```

---

## Exercises

1. Why is bias correction necessary in the first few steps of training? Compute m̂₁ with β₁=0.9 and g₁=0.5.
2. Show that if β₂→1, Adam approaches RMSprop. If β₁=0, it approaches AdaGrad.
3. For a parameter θ with consistently large gradients (large vₜ), is AdamW's effective learning rate larger or smaller than η? Why is this adaptive behaviour useful?
