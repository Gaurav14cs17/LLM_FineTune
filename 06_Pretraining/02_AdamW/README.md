# 02 — AdamW Optimiser

## 1. Adam — Full Algorithm with Proofs

```
MOTIVATION: SGD uses a single global learning rate for all parameters.
  Problem: some parameters have large gradients (need small LR to be stable)
           some parameters have small gradients (need large LR to learn)
  
  ADAM's solution: maintain per-parameter "effective learning rates" adapted from gradient history.

ALGORITHM (Kingma & Ba 2014):

  HYPERPARAMETERS: η (LR), β₁ (momentum decay), β₂ (variance decay), ε (stability constant)
  Typical: η=3×10⁻⁴, β₁=0.9, β₂=0.999, ε=10⁻⁸ (LLM training: β₂=0.95, ε=10⁻⁵)

  INITIALISE: m₀ = 0 (first moment), v₀ = 0 (second moment), t = 0

  FOR each step t = 1, 2, 3, ...:
    gₜ = ∂L/∂θ                           ← compute gradient

    mₜ = β₁ · m_{t-1} + (1 − β₁) · gₜ   ← exponentially weighted average of gradients
    vₜ = β₂ · v_{t-1} + (1 − β₂) · gₜ²  ← exponentially weighted average of squared grads
    
    m̂ₜ = mₜ / (1 − β₁ᵗ)                 ← bias correction
    v̂ₜ = vₜ / (1 − β₂ᵗ)                 ← bias correction
    
    θₜ = θ_{t-1} − η × m̂ₜ / (√v̂ₜ + ε)   ← parameter update
```

## 2. Bias Correction — Proof

```
PROBLEM: at t=1 with m₀=0:
  m₁ = β₁·0 + (1−β₁)·g₁ = (1−β₁)·g₁

  If β₁ = 0.9:  m₁ = 0.1·g₁   (much smaller than g₁!)
  → the estimate is BIASED toward 0 at early steps

DERIVATION OF BIAS:
  mₜ = (1−β₁) ∑_{i=1}^{t} β₁^{t-i} · gᵢ

  Expected value (assuming gᵢ ≈ g for all i, i.e., stationary):
    E[mₜ] = (1−β₁) ∑_{i=1}^{t} β₁^{t-i} · E[g]
           = (1−β₁) · E[g] · ∑_{i=1}^{t} β₁^{t-i}
           = (1−β₁) · E[g] · (1−β₁ᵗ)/(1−β₁)   ← geometric series
           = E[g] · (1−β₁ᵗ)

  We WANT E[m̂ₜ] = E[g].
  Bias correction: m̂ₜ = mₜ / (1−β₁ᵗ)
  E[m̂ₜ] = E[mₜ] / (1−β₁ᵗ) = E[g] · (1−β₁ᵗ) / (1−β₁ᵗ) = E[g]  ✓

HOW FAST DOES BIAS VANISH?
  t=1:  1−β₁¹ = 1−0.9 = 0.1  (90% bias — very biased!)
  t=5:  1−β₁⁵ = 1−0.59 = 0.41
  t=10: 1−β₁¹⁰ = 1−0.35 = 0.65
  t=50: 1−β₁⁵⁰ = 1−0.005 = 0.995  (bias almost gone)
  
  Effective warm-up: first ~50 steps are biased; correction essential.
```

## 3. AdamW — Decoupled Weight Decay

```
ORIGINAL L2 REGULARISATION:
  min_θ L(θ) + λ/2 × ‖θ‖²

  Adding L2 to the loss → gradient:
    ∇_θ [L(θ) + λ/2‖θ‖²] = gₜ + λ·θ
  
  Adam uses this combined gradient:
    vₜ = β₂ v_{t-1} + (1−β₂)(gₜ + λθ)²
    → The weight decay λθ ENTERS THE ADAPTIVE SCALING (√v̂)!
    → Highly regularised parameters (large θ) have inflated vₜ → smaller effective LR
    → Weight decay is less effective for frequently updated params!

ADAMW DECOUPLED WEIGHT DECAY (Loshchilov & Hutter 2017):
  Decouple weight decay from gradient adaptation:
  
  Step 1 (gradient step with Adam):
    m̂ₜ = bias_corrected first moment of gₜ   (NOT gₜ + λθ!)
    v̂ₜ = bias_corrected second moment of gₜ
    θ'  = θ − η × m̂ₜ / (√v̂ₜ + ε)            ← update from gradient
  
  Step 2 (weight decay step):
    θₜ = θ' − η · λ · θ_{t-1}               ← direct weight shrinkage
       = θ' × (1 − η·λ)                     ← equivalent formulation

  ┌──────────────────────────────────────────────────────────────┐
  │  θₜ = θ_{t-1} − η × m̂ₜ / (√v̂ₜ + ε) − η·λ·θ_{t-1}         │
  │                 ↑ gradient descent        ↑ weight decay     │
  │                 (adaptive)               (direct, NOT adaptive)│
  └──────────────────────────────────────────────────────────────┘

WHY BETTER?
  Standard Adam with L2: weight decay is modulated by v̂ (inconsistent regularisation)
  AdamW: weight decay is constant magnitude for all parameters (consistent regularisation)
  
  Empirically: AdamW outperforms Adam+L2 by 1-3% on LM benchmarks.
  Now the de-facto standard for LLM training.
```

## 4. Gradient Clipping — Why and How

```
PROBLEM: exploding gradients during training.
  Occasionally, a batch of tokens has unusually high loss.
  The gradient ‖g‖ can spike to 100× normal magnitude.
  → Single large update can destroy the model!

GRADIENT CLIPPING (standard in LLM training):
  If ‖g‖₂ > γ:
    g_clipped = g × γ / ‖g‖₂
  Else:
    g_clipped = g

  This RESCALES the gradient vector to have magnitude at most γ.
  DIRECTION is preserved; only magnitude is clipped.

TYPICAL VALUE: γ = 1.0 for most LLM training.

MATHEMATICAL EFFECT:
  Clipped gradient update magnitude:
    ‖Δθ‖ ≤ η × γ / ε    (bounded by clipping threshold / stability constant)
  
  For η=3e-4, γ=1, ε=1e-5: ‖Δθ‖ ≤ 3e-4 / 1e-5 = 30 (per-parameter)
  
  Wait — the per-parameter update is m̂/(√v̂+ε), not just g.
  With clipping, we're actually clipping ∑_i (∂L/∂θᵢ)².
  If a single gradient ∂L/∂θᵢ is too large, the ENTIRE gradient vector is rescaled.

WHEN DOES CLIPPING ACTIVATE?
  Plot ‖g‖ vs training step:
    Normal training:   ‖g‖ ∈ [0.5, 2.0]    (rarely clips)
    Bad batches:       ‖g‖ ∈ [5, 50]        (clips to 1.0)
    Early training:    ‖g‖ can be large      (clips frequently)
  
  Frequent clipping (>10% of steps) indicates:
    LR too high, or model instability, or data quality issues.
```

## 5. Learning Rate Schedules — Mathematical Forms

```
COSINE SCHEDULE (most common for LLMs):
  At training step t with T_max total steps:
  
  WARMUP PHASE (0 ≤ t ≤ T_warmup):
    η(t) = η_max × t / T_warmup   (linear ramp from 0 to η_max)
  
  COSINE DECAY PHASE (T_warmup < t ≤ T_max):
    η(t) = η_min + (η_max − η_min) × [1 + cos(π × (t−T_warmup)/(T_max−T_warmup))] / 2
  
  Shape:
    ↑ η_max ────────────────────────────────────────────────────────────────
    │          ╭──────────╮
    │         ╱           ╲
    │        ╱             ╲
    │       ╱               ╲
    │      ╱                 ╲
    │    ╱                    ╲──────────────────────────────── (floor η_min)
    └──────────────────────────────────────────────────────────────────────→ t
         ↑ warmup ↑            ↑ cosine decay
    t=0       T_warmup         T_max

WARMUP ANALYSIS:
  Without warmup: at t=1, Adam has m₀=v₀=0 → biased estimates.
    m̂ / (√v̂ + ε) is unstable initially.
  With linear warmup: start from small η → Adam's bias is small when η is small.
    By the time η reaches η_max, Adam has seen T_warmup steps → estimates are stable.
  
  TYPICAL: T_warmup = 1000-2000 steps (≈0.1-0.5% of T_max)

WHY COSINE OVER LINEAR DECAY?
  Linear: η(t) = η_max × (1 − t/T_max)  → fast initial drop, too slow later
  Cosine: slow initial drop → long high-LR phase → faster convergence
          slow final phase → allows fine-grained convergence near minimum

LEARNING RATE vs BATCH SIZE SCALING:
  Linear scaling rule (Goyal et al.): if batch_size is k× larger → multiply η by k.
  INTUITION: with k× more samples per step, each step provides k× more information,
  so we can afford k× larger steps.
  
  For LLaMA-3 training:
    Batch size ≈ 4M tokens  (vs typical 1K example batch)
    LR ≈ 3×10⁻⁴  (very large relative to SGD conventions)
```

---

## Exercises

1. Derive the bias in E[m₁] for β₁=0.9 without bias correction. By what factor is E[m₁] smaller than E[g]?
2. Show algebraically that L2 regularisation with Adam is NOT equivalent to AdamW. Specifically, show that the effective weight decay per unit of weight differs.
3. For a cosine schedule with η_max=3e-4, η_min=3e-5, T_warmup=2000, T_max=200000: compute η at steps t=0, 100, 2000, 10000, 100000, 200000.
