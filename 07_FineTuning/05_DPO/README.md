# Understanding DPO: Direct Preference Optimization

*From the RL objective to a simple binary cross-entropy loss — complete mathematical derivation*

---

**DPO (Direct Preference Optimization)** eliminates the need for reinforcement learning in RLHF by deriving a closed-form supervised loss from the RLHF objective. Introduced by Rafailov et al. (2023), DPO requires only a reference model and preference pairs — no reward model, no RL loop.

This guide derives DPO from scratch, proving each step.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 RLHF vs DPO](#11-rlhf-vs-dpo)
   - [1.2 What DPO Needs](#12-what-dpo-needs)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [RLHF Objective](#2-rlhf-objective)
   - [2.1 Formal Setup](#21-formal-setup)
   - [2.2 Bradley-Terry Preference Model](#22-bradley-terry-preference-model)
3. [Deriving the Optimal Policy](#3-deriving-the-optimal-policy)
   - [3.1 Closed-Form Solution](#31-closed-form-solution)
   - [3.2 The Key Rearrangement](#32-the-key-rearrangement)
4. [The DPO Loss](#4-the-dpo-loss)
   - [4.1 Z(x) Cancellation — The Core Insight](#41-zx-cancellation--the-core-insight)
   - [4.2 Full DPO Objective](#42-full-dpo-objective)
5. [DPO Gradient Analysis](#5-dpo-gradient-analysis)
   - [5.1 What DPO Updates](#51-what-dpo-updates)
6. [Implementation Details](#6-implementation-details)
   - [6.1 Computing Log-Ratios](#61-computing-log-ratios)
   - [6.2 Choosing β](#62-choosing-β)
7. [Summary](#7-summary)
   - [7.1 All Formulas Quick Reference](#71-all-formulas-quick-reference)
   - [7.2 RLHF vs DPO Comparison](#72-rlhf-vs-dpo-comparison)
   - [7.3 Common Mistakes](#73-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 RLHF vs DPO

```
┌─────────────────────────────────────────────────────────────┐
│  RLHF PIPELINE (complex):                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Preference data D = {(x, y_w, y_l)}                       │
│      ↓                                                      │
│  1. Train reward model r_φ(x, y) on D                       │
│      ↓                                                      │
│  2. Run PPO RL loop:                                        │
│     for each step:                                          │
│       sample y ~ π_θ(·|x)                                   │
│       compute r = r_φ(x, y) − β×KL(π_θ‖π_ref)              │
│       update π_θ via PPO gradient                           │
│                                                             │
│  PROBLEMS: 4 models in memory, unstable RL training,        │
│            reward hacking, hyperparameter sensitivity       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  DPO PIPELINE (simple):                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Preference data D = {(x, y_w, y_l)}                       │
│      ↓                                                      │
│  1. Keep π_ref = π_SFT frozen                               │
│  2. Minimise one loss function on D:                        │
│     L_DPO = −E[log σ(β log π_θ(y_w)/π_ref(y_w) −          │
│                       β log π_θ(y_l)/π_ref(y_l) )]         │
│                                                             │
│  Just TWO MODELS in memory.  Supervised learning only!      │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: RLHF is like hiring a manager (reward model) to grade essays, then using a complicated feedback loop to train the writer. DPO skips the manager and directly trains the writer by showing pairs of good vs bad essays.

### 1.2 What DPO Needs

| Requirement | RLHF | DPO |
|-------------|------|-----|
| Reward model | ✅ (must train) | ❌ (not needed) |
| Reference model π_ref | ✅ | ✅ |
| RL training loop | ✅ (PPO) | ❌ |
| Preference pairs (x, y_w, y_l) | ✅ | ✅ |
| Models in memory | 4 | 2 |
| Training stability | Lower | Higher |

### 1.3 Pipeline Summary

```
STEP 1: Collect preference data D = {(x, y_w, y_l)}
        ↓
STEP 2: Load π_ref = π_SFT (frozen copy of SFT model)
        ↓
STEP 3: For each batch from D:
          Compute: log π_θ(y_w|x), log π_θ(y_l|x)  [forward through trainable model]
          Compute: log π_ref(y_w|x), log π_ref(y_l|x)  [forward through frozen ref]
          Compute: DPO loss (see formula below)
        ↓
STEP 4: Update π_θ via gradient of DPO loss
        (π_ref always stays frozen)
        ↓
OUTPUT: Aligned model π_θ  (no reward model needed!)
```

---

## 2. RLHF Objective

### 2.1 Formal Setup

```
GOAL: Find a policy π_θ that:
  1. Maximises reward (human preference)
  2. Stays close to the reference policy π_ref (doesn't go off the rails)

RLHF OBJECTIVE:
  max_θ  J(θ) = E_{x~ρ, y~π_θ(·|x)} [r*(x,y)] − β × KL(π_θ(·|x) ‖ π_ref(·|x))
                ↑ maximise expected reward      ↑ KL penalty to stay near reference

  β > 0 controls the trade-off:
    Small β → freely maximise reward → risk of reward hacking
    Large β → stay close to π_ref → reward barely improves
```

### 2.2 Bradley-Terry Preference Model

```
ASSUMPTION: Human preferences come from a latent reward r*(x, y):

  P(y_w ≻ y_l | x) = σ(r*(x, y_w) − r*(x, y_l))
  
  where σ(z) = 1/(1 + e^{-z}) is the sigmoid function.
  
  INTUITION:
  ┌────────────────────────────────────────────────────────────┐
  │  r*(x, y_w) − r*(x, y_l) = 5   → very strong preference  │
  │    σ(5) = 0.993  ← 99.3% chance y_w is preferred          │
  │                                                            │
  │  r*(x, y_w) − r*(x, y_l) = 0   → no preference            │
  │    σ(0) = 0.5  ← coin flip                                │
  │                                                            │
  │  r*(x, y_w) − r*(x, y_l) = -2  → reverse preference      │
  │    σ(-2) = 0.119  ← 88% chance y_l is actually preferred  │
  └────────────────────────────────────────────────────────────┘
```

---

## 3. Deriving the Optimal Policy

### 3.1 Closed-Form Solution

```
┌─────────────────────────────────────────────────────────────┐
│  THEOREM: The optimal policy for the RLHF objective is:     │
│                                                             │
│  π*(y|x) = π_ref(y|x) × exp(r*(x,y)/β) / Z(x)             │
│                                                             │
│  where Z(x) = Σᵧ π_ref(y|x) × exp(r*(x,y)/β)             │
│               is the partition function (normaliser)        │
└─────────────────────────────────────────────────────────────┘

DERIVATION:
  Rewrite J(θ) with KL expansion:
    J(θ) = E_x E_{y~π_θ} [r*(x,y)] − β × E_x [KL(π_θ ‖ π_ref)]
         = −β × E_x [KL(π_θ(·|x) ‖ π_ref(·|x) × exp(r*(x,·)/β) / Z(x))] + const

  The KL divergence KL(p ‖ q) ≥ 0, with equality when p = q.
  
  Maximising J(θ) = minimising the KL
  → J* achieved when π_θ = π_ref × exp(r*/β) / Z  ■

VERIFICATION: Check it's a valid distribution:
  Σᵧ π*(y|x) = Σᵧ π_ref(y|x) × exp(r*(x,y)/β) / Z(x)
              = (1/Z) × Σᵧ π_ref(y|x) exp(r*/β)
              = Z/Z = 1 ✓
```

### 3.2 The Key Rearrangement

```
From the optimal policy equation:
  π*(y|x) = π_ref(y|x) × exp(r*(x,y)/β) / Z(x)

Take the log:
  log π*(y|x) = log π_ref(y|x) + r*(x,y)/β − log Z(x)

Rearrange to solve for r*:

  ┌──────────────────────────────────────────────────────────┐
  │  r*(x,y) = β × log[π*(y|x) / π_ref(y|x)] + β × log Z(x)│
  │              ─────────────────────────────   ─────────── │
  │               log-ratio (tractable!)          f(x) only  │
  └──────────────────────────────────────────────────────────┘

KEY INSIGHT: The reward can be expressed as a LOG-RATIO of policies!
  And the Z(x) term depends only on x, not on y.
```

---

## 4. The DPO Loss

### 4.1 Z(x) Cancellation — The Core Insight

```
┌─────────────────────────────────────────────────────────────┐
│  SUBSTITUTING INTO BRADLEY-TERRY:                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  P(y_w ≻ y_l | x) = σ(r*(x,y_w) − r*(x,y_l))              │
│                                                             │
│  r*(x,y_w) − r*(x,y_l)                                     │
│  = [β log(π*(y_w)/π_ref(y_w)) + β log Z(x)]                │
│  − [β log(π*(y_l)/π_ref(y_l)) + β log Z(x)]                │
│                                                             │
│  = β log[π*(y_w)/π_ref(y_w)] − β log[π*(y_l)/π_ref(y_l)]  │
│                                                             │
│  ← THE β log Z(x) TERMS CANCEL! (they're equal!) ←         │
│                                                             │
│  This is the MAGIC of DPO: the intractable Z(x) cancels!  │
└─────────────────────────────────────────────────────────────┘

So: P(y_w ≻ y_l | x) = σ( β log[π*(y_w|x)/π_ref(y_w|x)]
                          − β log[π*(y_l|x)/π_ref(y_l|x)] )
```

### 4.2 Full DPO Objective

Replacing the theoretical optimal π* with our trainable π_θ:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  L_DPO(θ) = −E_{(x,y_w,y_l)~D}  [log σ(                    │
│               β × log[π_θ(y_w|x) / π_ref(y_w|x)]           │
│             − β × log[π_θ(y_l|x) / π_ref(y_l|x)]           │
│             ) ]                                              │
│                                                              │
│  This is a BINARY CROSS-ENTROPY loss!                        │
│  No RL, no reward model, no PPO needed.                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘

DECOMPOSITION:
  Define:
    r_w = log π_θ(y_w|x) − log π_ref(y_w|x)   (implicit reward of winner)
    r_l = log π_θ(y_l|x) − log π_ref(y_l|x)   (implicit reward of loser)
  
  L_DPO = −E[ log σ(β(r_w − r_l)) ]
  
  The model is trained to make β(r_w − r_l) > 0:
  → make π_θ(y_w)/π_ref(y_w) large  (increase winner probability relative to ref)
  → make π_θ(y_l)/π_ref(y_l) small  (decrease loser probability relative to ref)
```

---

## 5. DPO Gradient Analysis

### 5.1 What DPO Updates

```
GRADIENT of L_DPO w.r.t. θ:

  Let u = β(r_w − r_l) = β × [(log π_θ(y_w) − log π_ref(y_w))
                              − (log π_θ(y_l) − log π_ref(y_l))]

  ∂L_DPO/∂u = −(1 − σ(u)) = −σ(−u)   [derivative of log-sigmoid]
  
  ∂u/∂θ = β × [∂log π_θ(y_w)/∂θ − ∂log π_θ(y_l)/∂θ]
         = β × [∇_θ log π_θ(y_w) − ∇_θ log π_θ(y_l)]

FULL GRADIENT:
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  ∂L_DPO/∂θ = −β × σ(−u) ×                                │
  │              [∇_θ log π_θ(y_w|x) − ∇_θ log π_θ(y_l|x)]  │
  │                                                            │
  │  ↑ correction    ↑ increase prob of y_w  ↓ decrease prob y_l│
  └────────────────────────────────────────────────────────────┘

CORRECTION FACTOR σ(−u) = 1 − σ(u):

  ┌────────────────────────────────────────────────────────────┐
  │  u >> 0 (model already prefers y_w):                       │
  │    σ(−u) ≈ 0 → small gradient → don't waste effort ✓     │
  │                                                            │
  │  u ≈ 0  (model is uncertain):                             │
  │    σ(−u) = 0.5 → moderate gradient → learn confidently   │
  │                                                            │
  │  u << 0 (model wrongly prefers y_l!):                     │
  │    σ(−u) ≈ 1 → LARGE gradient → fix urgently! ✓          │
  └────────────────────────────────────────────────────────────┘
  
  DPO automatically focuses learning on MISRANKED pairs.
```

---

## 6. Implementation Details

### 6.1 Computing Log-Ratios

```
For response y = (y₁, y₂, ..., y_T):

  log π_θ(y|x) = Σₜ₌₁ᵀ log P_θ(yₜ | x, y₁,...,yₜ₋₁)
               = sum of token-level log-probabilities

  log-ratio = log π_θ(y|x) − log π_ref(y|x)
            = Σₜ [log P_θ(yₜ|·) − log P_ref(yₜ|·)]

TRAINING COMPUTATION per batch:
  Step 1: Forward π_θ on (x, y_w) → log_π_θ_yw
  Step 2: Forward π_θ on (x, y_l) → log_π_θ_yl
  Step 3: Forward π_ref on (x, y_w) → log_π_ref_yw  [NO GRADIENT]
  Step 4: Forward π_ref on (x, y_l) → log_π_ref_yl  [NO GRADIENT]
  Step 5: ratio_w = log_π_θ_yw − log_π_ref_yw
          ratio_l = log_π_θ_yl − log_π_ref_yl
  Step 6: logit = β × (ratio_w − ratio_l)
  Step 7: loss = −log σ(logit)
  Step 8: Backward through π_θ only (steps 1 and 2)

MEMORY: need π_θ (trainable) + π_ref (frozen) in memory.
  For LLaMA-3 8B BF16: 2 × 16 GB = 32 GB minimum.
  In practice: load π_ref in 8-bit to save ~8 GB.
```

### 6.2 Choosing β

```
┌─────────────────────────────────────────────────────────────┐
│  β CONTROLS: HOW MUCH TO DEVIATE FROM π_ref                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SMALL β (e.g., 0.1):                                       │
│  + Can explore far from π_ref → large alignment improvement │
│  - Risk of reward hacking / degenerate outputs              │
│                                                             │
│  LARGE β (e.g., 1.0):                                       │
│  + Safe, stays close to π_ref                               │
│  - Alignment barely improves                                │
│                                                             │
│  MONITORING:                                                │
│  ✓ GOOD: reward margin r_w − r_l increasing over training  │
│  ✓ GOOD: KL(π_θ ‖ π_ref) stays bounded (< 10 nats)        │
│  ✗ BAD:  KL growing unboundedly → decrease β               │
│  ✗ BAD:  reward barely improving → decrease β              │
│                                                             │
│  EMPIRICAL RECOMMENDATIONS:                                 │
│    π_ref = base model → fine-tuning heavily: β = 0.5       │
│    π_ref = SFT model  → chat alignment:      β = 0.1       │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Summary

### 7.1 All Formulas Quick Reference

**RLHF Objective:**

```
max_θ  J(θ) = E[r*(x,y)] − β × KL(π_θ ‖ π_ref)
```

**Optimal Policy:**

```
π*(y|x) = π_ref(y|x) × exp(r*(x,y)/β) / Z(x)
```

**Reward in Terms of Policy:**

```
r*(x,y) = β × log[π*(y|x) / π_ref(y|x)] + β × log Z(x)
```

**DPO Loss:**

```
L_DPO = −E[log σ( β × log[π_θ(y_w)/π_ref(y_w)] − β × log[π_θ(y_l)/π_ref(y_l)] )]
```

### 7.2 RLHF vs DPO Comparison

| Aspect | RLHF-PPO | DPO |
|--------|----------|-----|
| Training type | Reinforcement Learning | Supervised learning |
| Models needed | 4 (policy, ref, reward, value) | 2 (policy, ref) |
| Training stability | Lower (RL is noisy) | Higher |
| Reward model | Separate training required | Implicit (no separate model) |
| Computation | ~3× SFT per step | ~2× SFT per step |
| Hyperparameters | Many (clip ε, λ, β, LR, ...) | Few (β, LR) |
| Reward hacking | Possible | Less likely (bounded by KL) |

### 7.3 Common Mistakes

```
❌ WRONG: Applying gradients to π_ref
✓ RIGHT:  π_ref is ALWAYS frozen. It provides the baseline.
          Only π_θ receives gradient updates.

❌ WRONG: Forgetting the β scaling factor in the loss
✓ RIGHT:  The loss is −log σ(β × (r_w − r_l)).
          Without β: the magnitude of the logit has no connection to KL.
          β controls how "strong" the learning signal is relative to KL cost.

❌ WRONG: Computing log-ratio over the full sequence including the prompt
✓ RIGHT:  Only compute log-probabilities over RESPONSE tokens.
          The prompt tokens are shared and cancel in the ratio anyway.

❌ WRONG: Using DPO for tasks with a clear single correct answer (e.g., math)
✓ RIGHT:  DPO is for preference-based alignment, not factual correctness.
          For math/code, supervised fine-tuning on verified solutions is better.
```

---

## 8. Exercises

1. **Z(x) Cancellation**: Verify algebraically that log Z(x) cancels when computing r*(y_w) − r*(y_l). Why does this only work because both terms have the SAME x?

2. **Gradient Direction**: For β=0.1 and current model state where u = β(r_w−r_l) = −2 (model wrongly prefers y_l): compute the correction factor σ(−u). Is the gradient large or small? What does this mean for learning?

3. **DPO vs SFT**: If you have a dataset of preference pairs (x, y_w, y_l), could you instead just do SFT on (x, y_w) only? What information would you lose? Under what conditions would SFT-only be sufficient?

4. **Memory Budget**: For DPO training of LLaMA-3 8B with π_ref in INT8 and π_θ in BF16 with LoRA (r=16): estimate the total memory required. Compare to full DPO with both models in BF16.
