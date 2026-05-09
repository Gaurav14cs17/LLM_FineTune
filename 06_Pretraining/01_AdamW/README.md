# Understanding AdamW: Adaptive Moment Optimizer with Weight Decay

*From SGD to Adam to AdamW — complete derivation, bias correction proof, and LLM training tips*

---

**AdamW** is the optimizer used to train virtually every large language model — GPT, LLaMA, PaLM, Mistral. It combines adaptive per-parameter learning rates (Adam) with proper weight decay that doesn't interfere with the momentum statistics. Understanding AdamW deeply explains why LLM training needs careful hyperparameter tuning.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Not Vanilla SGD?](#11-why-not-vanilla-sgd)
   - [1.2 The Evolution: SGD → Adam → AdamW](#12-the-evolution-sgd--adam--adamw)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Adam: Adaptive Learning Rates](#2-adam-adaptive-learning-rates)
   - [2.1 The Algorithm](#21-the-algorithm)
   - [2.2 Bias Correction — Full Proof](#22-bias-correction--full-proof)
3. [AdamW: Weight Decay Fix](#3-adamw-weight-decay-fix)
   - [3.1 The Problem with Adam's L2 Regularisation](#31-the-problem-with-adams-l2-regularisation)
   - [3.2 Decoupled Weight Decay Derivation](#32-decoupled-weight-decay-derivation)
4. [Gradient Clipping](#4-gradient-clipping)
5. [Learning Rate Schedules](#5-learning-rate-schedules)
   - [5.1 Cosine Annealing with Warmup](#51-cosine-annealing-with-warmup)
   - [5.2 Why Warmup?](#52-why-warmup)
6. [Memory Layout](#6-memory-layout)
7. [Summary](#7-summary)
   - [7.1 All Formulas Quick Reference](#71-all-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 Why Not Vanilla SGD?

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Parameter scales differ wildly in LLMs            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EXAMPLE: LLaMA-3 8B at training step 0                     │
│                                                             │
│  embedding weight W_E:  gradient ≈ 0.01  (sparse updates)  │
│  attention W_Q:         gradient ≈ 0.001 (dense, small)     │
│  FFN gate:              gradient ≈ 0.1   (much larger!)     │
│                                                             │
│  SGD with lr=0.001:                                         │
│  W_E update: 0.001 × 0.01  = 1e-5   ← too slow to learn    │
│  FFN update: 0.001 × 0.1   = 1e-4   ← maybe ok             │
│                                                             │
│  If we increase lr to help W_E: FFN updates become 0.1     │
│  → FFN diverges!                                            │
│                                                             │
│  ADAM SOLUTION: estimate per-parameter "scale" from history │
│  and normalise each gradient by its own magnitude estimate. │
│  Every parameter gets its own effective learning rate!      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Evolution: SGD → Adam → AdamW

```
SGD:     θₜ = θₜ₋₁ − η × gₜ
           ↓ problem: same LR for all params

Adam:    mₜ = β₁mₜ₋₁ + (1−β₁)gₜ           (1st moment = mean)
         vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²          (2nd moment = variance)
         m̂ₜ = mₜ/(1−β₁ᵗ),  v̂ₜ = vₜ/(1−β₂ᵗ)   (bias correction)
         θₜ = θₜ₋₁ − η × m̂ₜ / (√v̂ₜ + ε)  ← adaptive!
           ↓ problem: L2 penalty mixed into adaptive statistics

AdamW:   Same as Adam, PLUS:
         θₜ = θₜ₋₁ − η × m̂ₜ/(√v̂ₜ+ε) − η×λ×θₜ₋₁   ← SEPARATE weight decay!
```

### 1.3 Pipeline Summary

```
For each training step t:
  
  1. Compute gradient: g_t = ∇_θ L(θ_t-1; batch_t)
       ↓
  2. Clip gradient: ĝ_t = g_t × min(1, c / ‖g_t‖)
       ↓
  3. Update moments:
     m_t = β₁ × m_{t-1} + (1-β₁) × ĝ_t
     v_t = β₂ × v_{t-1} + (1-β₂) × ĝ_t²
       ↓
  4. Bias-correct moments:
     m̂_t = m_t / (1 - β₁ᵗ)
     v̂_t = v_t / (1 - β₂ᵗ)
       ↓
  5. Compute current LR: η_t = η_peak × lr_schedule(t)
       ↓
  6. Update parameters:
     θ_t = θ_{t-1} × (1 - η_t × λ)  ← weight decay
          - η_t × m̂_t / (√v̂_t + ε) ← adaptive gradient step
```

---

## 2. Adam: Adaptive Learning Rates

### 2.1 The Algorithm

```
HYPERPARAMETERS:
  η     = 3×10⁻⁴   (peak learning rate, typical for LLMs)
  β₁    = 0.9       (1st moment decay — "momentum")
  β₂    = 0.95      (2nd moment decay — "adaptive scale")
  ε     = 1×10⁻⁸   (numerical stability)

INITIALISE: m₀ = 0, v₀ = 0

AT EACH STEP t (from 1 to T):

  Step 2a: mₜ = β₁ × mₜ₋₁ + (1 − β₁) × gₜ
  Step 2b: vₜ = β₂ × vₜ₋₁ + (1 − β₂) × gₜ²

  Step 3a: m̂ₜ = mₜ / (1 − β₁ᵗ)
  Step 3b: v̂ₜ = vₜ / (1 − β₂ᵗ)

  Step 4:  θₜ = θₜ₋₁ − η × m̂ₜ / (√v̂ₜ + ε)

INTUITION FOR EACH COMPONENT:
  mₜ: Exponential moving average of past gradients.
      Tracks gradient DIRECTION — reduces noise.
      β₁=0.9 → "past 10 steps" worth of gradient averaging.

  vₜ: Exponential moving average of squared gradients.
      Tracks gradient SCALE — estimates typical magnitude.
      β₂=0.95 → "past 20 steps" worth of magnitude averaging.

  m̂ₜ/(√v̂ₜ+ε): "normalised gradient signal"
      ≈ gradient / (estimated std of gradients)
      ≈ z-score → roughly unit magnitude for every parameter!
```

### 2.2 Bias Correction — Full Proof

```
┌─────────────────────────────────────────────────────────────┐
│  WHY BIAS CORRECTION?                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  m₀ = 0 (initialised to zero)                               │
│                                                             │
│  At step 1:                                                 │
│    m₁ = β₁×m₀ + (1−β₁)×g₁ = 0 + (1−β₁)×g₁ = 0.1×g₁       │
│                                                             │
│    If g₁ = 1.0: m₁ = 0.1   ← WRONG! Should be ≈ 1.0       │
│                                                             │
│  At step 10 (steady state, all gₜ ≈ g):                    │
│    m₁₀ ≈ g × Σₖ₌₁¹⁰ β₁^{10-k} × (1−β₁)                   │
│         = g × (1−β₁) × (1−β₁¹⁰)/(1−β₁)                   │
│         = g × (1−β₁¹⁰) = g × 0.65  ← still biased low!    │
│                                                             │
│  BIAS CORRECTION PROOF:                                     │
│  The true estimate we want: E[mₜ] = g (the true gradient)  │
│                                                             │
│  mₜ = Σₖ₌₁ᵗ (1−β₁)β₁^{t-k} gₖ  + β₁ᵗ × m₀               │
│     = Σₖ₌₁ᵗ (1−β₁)β₁^{t-k} gₖ  (since m₀=0)              │
│                                                             │
│  E[mₜ] = g × Σₖ₌₁ᵗ (1−β₁)β₁^{t-k}  (assuming gₖ ≈ g)     │
│        = g × (1−β₁) × (1−β₁ᵗ)/(1−β₁)                     │
│        = g × (1−β₁ᵗ)                                       │
│                                                             │
│  CORRECTION:  m̂ₜ = mₜ / (1−β₁ᵗ)                           │
│  E[m̂ₜ] = E[mₜ] / (1−β₁ᵗ) = g × (1−β₁ᵗ) / (1−β₁ᵗ) = g ✓ │
│                                                             │
│  Same proof applies to v̂ₜ = vₜ / (1−β₂ᵗ)                 │
└─────────────────────────────────────────────────────────────┘
```

**Numerical verification:**

```
β₁ = 0.9, constant gradient gₜ = 1.0 for all t:

  t=1:  m₁=0.100,  (1-β₁¹)=0.100,  m̂₁=1.000 ✓
  t=2:  m₂=0.190,  (1-β₁²)=0.190,  m̂₂=1.000 ✓
  t=5:  m₅=0.410,  (1-β₁⁵)=0.410,  m̂₅=1.000 ✓
  t=10: m₁₀=0.651, (1-β₁¹⁰)=0.651, m̂₁₀=1.000 ✓
  
  Bias correction perfectly recovers the true gradient at every step!
```

---

## 3. AdamW: Weight Decay Fix

### 3.1 The Problem with Adam's L2 Regularisation

```
┌─────────────────────────────────────────────────────────────┐
│  ADAM + L2 REGULARISATION (WRONG):                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L2 regularisation: loss_total = loss + (λ/2) × Σ θᵢ²     │
│  This adds λ×θ to the gradient: g_total = g + λ×θ          │
│                                                             │
│  PROBLEM: the L2 penalty gradient (λθ) enters the           │
│  1st and 2nd moment estimates:                              │
│                                                             │
│  mₜ = β₁mₜ₋₁ + (1-β₁)(gₜ + λθₜ)   ← λθ mixes into m!   │
│  vₜ = β₂vₜ₋₁ + (1-β₂)(gₜ + λθₜ)² ← λθ mixes into v!    │
│                                                             │
│  CONSEQUENCE: the adaptive normalisation m̂/(√v̂+ε)         │
│  interacts with the regularisation term unpredictably.     │
│  Effective weight decay ≠ λ × θ.                          │
│  It's λ×θ / (√v̂+ε) — the actual decay depends on         │
│  per-parameter gradient scale, which varies wildly!        │
│                                                             │
│  RESULT: Adam + L2 regularisation doesn't actually apply   │
│  the intended weight decay. Large-gradient parameters get  │
│  MORE regularisation, small-gradient ones get LESS.        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Decoupled Weight Decay Derivation

```
ADAMW SOLUTION: separate the weight decay from the gradient update.

ADAMW UPDATE RULE:

  m_t = β₁ × m_{t-1} + (1-β₁) × g_t      ← only gradient, no λθ!
  v_t = β₂ × v_{t-1} + (1-β₂) × g_t²     ← only gradient, no λθ!

  θ_t = θ_{t-1} - η × [m̂_t / (√v̂_t + ε)]     ← Adam step
        - η × λ × θ_{t-1}                       ← SEPARATE weight decay

INTERPRETATION:
  The weight decay η×λ×θ is applied AFTER the adaptive gradient step.
  It's a fixed fraction λ of the weight, regardless of gradient scale.

PARAMETER VALUE EVOLUTION (with weight decay, no gradient):
  θ_t = θ_{t-1} × (1 - η×λ)

  Per step: weight is multiplied by (1 - η×λ)
  
  For η=3×10⁻⁴, λ=0.1:
    (1 - η×λ) = (1 - 3×10⁻⁵) = 0.99997 per step
  
  After T=500,000 steps (LLaMA-3 8B training):
    0.99997^{500000} = exp(-15) ≈ 3×10⁻⁷  ← approaches zero!
    
  THIS IS INTENTIONAL: weights that don't contribute to predictions
  should decay to zero. Only useful weights survive.
```

---

## 4. Gradient Clipping

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Gradient spikes cause catastrophic parameter updates│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Normal step: ‖g‖ = 0.3 → update θ by ~0.001              │
│  Spike step:  ‖g‖ = 50  → update θ by ~0.15  ← disaster!  │
│                                                             │
│  These happen e.g. when:                                    │
│  - A batch contains unexpectedly long sequences             │
│  - A numerical instability in the forward pass              │
│  - Early in training when loss landscape is steep           │
└─────────────────────────────────────────────────────────────┘

GLOBAL GRADIENT CLIPPING:

  Compute global norm: ‖g‖ = √(Σᵢ Σⱼ gᵢⱼ²)   (over ALL parameters)
  
  If ‖g‖ > c_max (typical: c_max = 1.0):
    Scale ALL gradients by c_max/‖g‖:
    ĝ = g × (c_max / ‖g‖)

  This preserves the DIRECTION of the gradient but caps the magnitude.

VISUALISATION:
  g = [0.8, 0.6] → ‖g‖ = 1.0 ≤ 1.0 → no clipping  → update = [0.8, 0.6]
  g = [80,  60]  → ‖g‖ = 100 > 1.0 → scale by 0.01 → update = [0.8, 0.6]

  SAME direction, but spike is capped to safe magnitude!

WHY GLOBAL (not per-parameter)?
  Per-parameter clipping distorts the gradient direction.
  Global clipping scales all parameters equally, preserving direction.
```

---

## 5. Learning Rate Schedules

### 5.1 Cosine Annealing with Warmup

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  η_t = η_min + (1/2)(η_peak − η_min)(1 + cos(πt'/T_decay)) │
│                                                              │
│  where:                                                      │
│    t' = t − T_warmup   (steps after warmup ends)            │
│    T_decay = T − T_warmup  (total cosine decay steps)       │
│                                                              │
└──────────────────────────────────────────────────────────────┘

FULL SCHEDULE:

  ┌─────────────────────────────────────────────────────────┐
  │  η_peak                          ╭───────╮              │
  │                                 ╱         ╲             │
  │                               ╱             ╲           │
  │                             ╱                 ╲         │
  │                           ╱                    ╲        │
  │  η_min  ──────────────────                      ─────── │
  │         ↑Warmup starts  ↑Warmup ends              ↑End  │
  │         T_start    T_warmup                      T_total │
  └─────────────────────────────────────────────────────────┘

  Phase 1 (Linear warmup): t ≤ T_warmup
    η_t = η_peak × (t / T_warmup)

  Phase 2 (Cosine decay): t > T_warmup
    η_t = η_min + (η_peak − η_min)/2 × [1 + cos(π(t−T_warmup)/(T−T_warmup))]

TYPICAL VALUES (LLaMA-3 8B):
  η_peak  = 3×10⁻⁴
  η_min   = 3×10⁻⁵   (10% of peak)
  T_warmup = 2000 steps
  T_total  = 500000 steps
```

### 5.2 Why Warmup?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION: Bias Correction Is Not Enough at Step 1         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  At step t=1: v̂₁ = g₁²/(1-β₂) = g₁²/0.05                 │
│    √v̂₁ = g₁/√0.05 ≈ 4.5 × g₁                             │
│    Effective step = η × m̂₁/√v̂₁ = η × 1/4.5 × (g₁/g₁)    │
│    Step ≈ η/4.5  ← smaller than peak                        │
│                                                             │
│  BUT:  v̂ is a poor estimate from just 1 data point.        │
│  If one batch is unusually large (noisy): v̂₁ is too big    │
│  If one batch is unusually small: v̂₁ underestimates σ²     │
│  → Steps in wrong direction with high confidence!           │
│                                                             │
│  WARMUP: let vₜ accumulate over ~2000 steps before          │
│  using the full learning rate. By then v̂_t is a reliable  │
│  estimate of gradient variance, and steps are correct.     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Memory Layout

```
OPTIMIZER STATE MEMORY (per parameter value):
  FP32 master weight:  4 bytes  ← always in float32 for accuracy
  Adam 1st moment m:  4 bytes  ← float32
  Adam 2nd moment v:  4 bytes  ← float32
  Total:              12 bytes per parameter

FOR LLAMA-3 8B (8 × 10⁹ parameters):
  Optimizer states: 12 × 8B = 96 GB

COMBINED TRAINING MEMORY:
  ┌─────────────────────────────────────────────────────────┐
  │  BF16 model weights:  16 GB                             │
  │  BF16 gradients:      16 GB                             │
  │  FP32 master weights: 32 GB                             │
  │  FP32 m₁ moments:     32 GB                             │
  │  FP32 v₂ moments:     32 GB                             │
  │  Activations:         varies (10-50 GB)                  │
  │  Total:               138-178 GB                        │
  │  → Requires 2-4× A100/H100 GPUs (80 GB each)            │
  └─────────────────────────────────────────────────────────┘
```

---

## 7. Summary

### 7.1 All Formulas Quick Reference

**Adam Update:**

```
mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ           (m₀ = 0)
vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²          (v₀ = 0)
m̂ₜ = mₜ / (1-β₁ᵗ)                   (bias correction)
v̂ₜ = vₜ / (1-β₂ᵗ)                   (bias correction)
```

**AdamW Step:**

```
θₜ = θₜ₋₁ - ηₜ × m̂ₜ / (√v̂ₜ + ε) - ηₜ × λ × θₜ₋₁
              ↑ Adam gradient step      ↑ weight decay
```

**Gradient Clipping:**

```
if ‖g‖ > c:  g ← g × c/‖g‖   (preserves direction)
```

**Cosine LR with Warmup:**

```
t ≤ T_w: η_t = η_peak × t/T_w
t > T_w: η_t = η_min + (η_peak-η_min)/2 × (1 + cos(π(t-T_w)/(T-T_w)))
```

| Hyperparameter | LLaMA-3 8B value | Effect if too high | Effect if too low |
|----------------|------------------|-------------------|------------------|
| η_peak | 3×10⁻⁴ | Divergence/spikes | Very slow learning |
| β₁ | 0.9 | Noisy, slow | Ignores momentum |
| β₂ | 0.95 | Adapts slowly | Adapts too fast |
| λ (weight decay) | 0.1 | Weights shrink to 0 | Overfitting |
| c (grad clip) | 1.0 | Gradient spikes | Steps too small |

### 7.2 Common Mistakes

```
❌ WRONG: Adam and AdamW are the same when λ=0
✓ RIGHT:  True — when λ=0, weight decay term is 0. The difference only
          matters when you want actual weight regularisation.

❌ WRONG: High β₂ = 0.999 is always better for LLM training
✓ RIGHT:  LLMs often use β₂=0.95 (faster adaptation than BERT's 0.999).
          Lower β₂ means v_t adapts faster to changing gradient scales,
          which matters during the rapid learning of early training.

❌ WRONG: Bias correction is purely for numerical stability
✓ RIGHT:  Bias correction is necessary for CORRECT gradient estimates.
          Without it, the first few steps use LR effectively 10-20× too small
          (β₁=0.9 → m₁ = 0.1×g₁ → only 10% of gradient is used initially).

❌ WRONG: Weight decay and L2 regularisation are the same in Adam
✓ RIGHT:  They are equivalent in SGD but NOT in Adam. In Adam, L2
          regularisation changes the gradient → enters moment statistics.
          Weight decay in AdamW directly modifies the parameter, not the gradient.
```

---

## 8. Exercises

1. **Bias Correction**: For β₁=0.9, constant gradient g=2.0, compute m₁, m₂, m₃ and their bias-corrected versions m̂₁, m̂₂, m̂₃. At what step t does m̂ₜ reach within 5% of g?

2. **AdamW vs Adam+L2**: For a parameter θ=0.5, gradient g=0.1, v̂=0.01, m̂=0.05, η=0.001, λ=0.01: compute the update for Adam+L2 and AdamW. Show they give different results.

3. **LR Schedule**: For η_peak=3×10⁻⁴, η_min=3×10⁻⁵, T_warmup=2000, T=100000: compute η at t=1000, 2000, 51000, 100000. Plot the four values to sketch the schedule shape.

4. **Optimizer Memory**: For a model with 3 billion parameters, compute the total optimizer state memory in GB (FP32 for m, v, and master weights). Compare to BF16 model weights. What fraction of training memory is the optimizer?
