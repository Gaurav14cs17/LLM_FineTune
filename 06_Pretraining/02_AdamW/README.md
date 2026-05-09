# Understanding the AdamW Optimizer (Supplementary)

*A concise second reference on Adam/AdamW — bias correction derivation and weight decay*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Adam for LLMs?](#11-why-adam-for-llms)
2. [Adam Algorithm Recap](#2-adam-algorithm-recap)
   - [2.1 Moment Estimates](#21-moment-estimates)
   - [2.2 Bias Correction Derivation](#22-bias-correction-derivation)
3. [AdamW: Decoupled Weight Decay](#3-adamw-decoupled-weight-decay)
   - [3.1 L2 Regularisation vs Weight Decay](#31-l2-regularisation-vs-weight-decay)
   - [3.2 Why Decoupling Matters](#32-why-decoupling-matters)
4. [LLaMA-3 Training Config](#4-llama-3-training-config)
5. [Common Mistakes](#5-common-mistakes)
6. [Exercises](#6-exercises)

---

## 1. Overview

### 1.1 Why Adam for LLMs?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SGD: same learning rate for ALL parameters.                │
│    Embedding of "the" (updated every batch): LR is fine.    │
│    Embedding of "quixotic" (updated once/1000 batches):     │
│    Same LR → too small when it finally gets a gradient!     │
│                                                             │
│  Adam: ADAPTS learning rate per-parameter.                  │
│    Frequent params: larger denominator → smaller effective LR│
│    Rare params: smaller denominator → larger effective LR   │
│    → All params learn at similar rates despite frequency!   │
│                                                             │
│  MEMORY COST: Adam stores m (momentum) + v (variance):     │
│    2 × N × 4 bytes = 8N bytes extra (FP32)                 │
│    For 8B model: 64 GB just for optimizer states!           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Adam Algorithm Recap

### 2.1 Moment Estimates

```
ADAM UPDATE RULE:

  gₜ = ∇L(θₜ₋₁)                    ← gradient at step t
  mₜ = β₁ × mₜ₋₁ + (1-β₁) × gₜ    ← first moment (mean estimate)
  vₜ = β₂ × vₜ₋₁ + (1-β₂) × gₜ²   ← second moment (variance estimate)
  
  m̂ₜ = mₜ / (1 - β₁ᵗ)              ← bias-corrected mean
  v̂ₜ = vₜ / (1 - β₂ᵗ)              ← bias-corrected variance
  
  θₜ = θₜ₋₁ - η × m̂ₜ / (√v̂ₜ + ε)  ← parameter update

VARIABLES:
| Symbol | Meaning | LLaMA-3 value |
|--------|---------|--------------|
| η | Learning rate | 3×10⁻⁴ (peak) |
| β₁ | Momentum decay | 0.9 |
| β₂ | Variance decay | 0.95 |
| ε | Numerical stability | 10⁻⁸ |
| λ | Weight decay (AdamW) | 0.1 |
```

### 2.2 Bias Correction Derivation

```
WHY BIAS CORRECTION?

  At initialisation: m₀ = 0, v₀ = 0 (zero-initialized)
  
  After t steps:
    mₜ = (1-β₁) × Σₖ₌₁ᵗ β₁^{t-k} × gₖ
  
  Expected value:
    E[mₜ] = (1-β₁) × Σₖ₌₁ᵗ β₁^{t-k} × E[gₖ]
           = E[g] × (1-β₁) × (1-β₁ᵗ)/(1-β₁)  (geometric sum)
           = E[g] × (1-β₁ᵗ)
  
  BIASED! E[mₜ] = (1-β₁ᵗ) × E[g] ≠ E[g]
  
  Especially bad for early steps:
    t=1: E[m₁] = (1-0.9¹) × E[g] = 0.1 × E[g]  ← 10× too small!
    t=10: E[m₁₀] = (1-0.9¹⁰) × E[g] = 0.65 × E[g]  ← still biased
    t=100: E[m₁₀₀] = (1-0.9¹⁰⁰) × E[g] ≈ 1.0 × E[g]  ← converged
  
  CORRECTION: divide by (1-β₁ᵗ) → E[m̂ₜ] = E[g]  ✓
  Same logic applies to vₜ with β₂.
```

---

## 3. AdamW: Decoupled Weight Decay

### 3.1 L2 Regularisation vs Weight Decay

```
L2 REGULARISATION (wrong for Adam):
  L_total = L_data + (λ/2) × ‖θ‖²
  ∂L_total/∂θ = g + λ×θ
  
  With Adam: the λ×θ gradient gets DIVIDED by √v!
  If θ has large v (high variance): weight decay is weakened.
  If θ has small v: weight decay is amplified.
  → L2 interacts BADLY with adaptive learning rates.

WEIGHT DECAY (correct — AdamW):
  θₜ = θₜ₋₁ - η × m̂ₜ/(√v̂ₜ + ε) - η × λ × θₜ₋₁
                                      ↑ applied DIRECTLY to weights
                                        NOT through the gradient!
  
  Weight decay is INDEPENDENT of the adaptive scaling.
  Every parameter shrinks by the same factor η×λ per step.
```

### 3.2 Why Decoupling Matters

```
EFFECT ON DIFFERENT PARAMETERS:

  Parameter with large gradient variance (e.g., output head):
    Adam L2: decay × θ is divided by large √v → almost no decay!
    AdamW:   decay × θ applied directly → proper regularisation ✓

  Parameter with small gradient variance (e.g., bias terms):  
    Adam L2: decay × θ is divided by small √v → excessive decay!
    AdamW:   decay × θ applied directly → proper regularisation ✓

  RESULT: AdamW gives UNIFORM regularisation strength across all params.
  Loshchilov & Hutter (2019) showed this leads to better generalisation.
```

---

## 4. LLaMA-3 Training Config

```
LLAMA-3 8B OPTIMIZER SETTINGS:
  Optimizer:     AdamW
  β₁ = 0.9      (momentum)
  β₂ = 0.95     (RMS — note: lower than default 0.999 for stability)
  ε = 10⁻⁸
  λ = 0.1        (weight decay)
  η_max = 3×10⁻⁴ (peak learning rate)
  η_min = 3×10⁻⁵ (final learning rate, 10× lower)
  Warmup: 2000 steps linear
  Schedule: cosine decay from η_max to η_min
  Gradient clipping: max_norm = 1.0

MEMORY for 8B model (FP32 optimizer states):
  Weights (BF16):  16 GB
  Gradients (BF16): 16 GB
  Master (FP32):   32 GB
  m₁ (FP32):      32 GB
  m₂ (FP32):      32 GB
  TOTAL:          128 GB  ← needs multi-GPU!
```

---

## 5. Common Mistakes

```
❌ WRONG: AdamW = Adam + L2 regularisation
✓ RIGHT:  AdamW = Adam + DECOUPLED weight decay (applied directly to weights,
          NOT through the gradient). Adam + L2 gives wrong effective decay
          because the adaptive denominator √v interferes with the L2 gradient.

❌ WRONG: β₂=0.999 is always best (it's the default)
✓ RIGHT:  For LLM pretraining, β₂=0.95 works better than 0.999.
          Lower β₂ → faster adaptation to gradient statistics changes.
          0.999 responds too slowly when data distribution shifts during training.

❌ WRONG: Bias correction doesn't matter after warmup
✓ RIGHT:  For β₂=0.95: (1-β₂ᵗ) reaches 0.99 after t≈90 steps.
          For β₂=0.999: (1-β₂ᵗ) reaches 0.99 after t≈4600 steps!
          With learning rate warmup of 2000 steps, bias correction
          is essential for the entire warmup phase and beyond.

❌ WRONG: Gradient clipping replaces the need for careful LR tuning
✓ RIGHT:  Clipping prevents catastrophic spikes (loss → NaN) but doesn't
          optimise convergence speed. Too high clip → no protection.
          Too low clip → equivalent to reducing LR (but non-uniformly).
          Both clip threshold AND learning rate must be tuned together.
```

---

## 6. Exercises

1. **Bias Correction**: At step t=5 with β₁=0.9 and β₂=0.95: compute the bias correction factors (1-β₁ᵗ) and (1-β₂ᵗ). By what factor are uncorrected m and v underestimated?

2. **Memory Budget**: For a 70B model using AdamW with FP32 optimizer states: compute total optimizer memory. Compare to ZeRO-Stage-1 with 8 GPUs (only shard optimizer states).

3. **L2 vs Weight Decay**: Parameter θ=1.0 with gradient g=0.5 and running v=0.25. Compare the effective weight decay between: (a) Adam with L2 (λ=0.1) and (b) AdamW (λ=0.1). Show the numerical difference.

4. **Adaptive LR**: Two parameters: param_A has v=1.0, param_B has v=0.01. With base LR η=3e-4: compute the effective learning rate for each. What ratio of step sizes does Adam produce?
