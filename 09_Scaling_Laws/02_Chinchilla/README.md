# Understanding Chinchilla Scaling Laws

*The compute-optimal recipe for training large language models*

---

**Chinchilla (Hoffmann et al. 2022)** corrected a fundamental error in Kaplan's 2020 scaling laws: large models were being severely undertrained. The Chinchilla paper showed that for a fixed compute budget, you should train a *smaller* model on *much more data* — roughly 20 tokens per parameter.

This guide derives the Chinchilla formulas from scratch and explains the "overtrained model" strategy used by LLaMA.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What Changed from Kaplan?](#11-what-changed-from-kaplan)
   - [1.2 The Loss Formula](#12-the-loss-formula)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The Parametric Loss Model](#2-the-parametric-loss-model)
   - [2.1 Formula and Constants](#21-formula-and-constants)
   - [2.2 Numerical Examples](#22-numerical-examples)
3. [Deriving Optimal N and D](#3-deriving-optimal-n-and-d)
   - [3.1 Lagrange Multiplier Derivation](#31-lagrange-multiplier-derivation)
   - [3.2 The D/N ≈ 20 Rule](#32-the-dn--20-rule)
4. [The Overtrained Model Strategy](#4-the-overtrained-model-strategy)
   - [4.1 Training Cost vs Inference Cost](#41-training-cost-vs-inference-cost)
   - [4.2 LLaMA's Approach](#42-llamas-approach)
5. [Isoloss Contours](#5-isoloss-contours)
6. [Summary](#6-summary)
   - [6.1 Formulas Quick Reference](#61-formulas-quick-reference)
   - [6.2 Kaplan vs Chinchilla](#62-kaplan-vs-chinchilla)
   - [6.3 Common Mistakes](#63-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 What Changed from Kaplan?

```
┌─────────────────────────────────────────────────────────────┐
│  KAPLAN (2020): "Scale model size aggressively"             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For C = 10²³ FLOPs:                                        │
│    N* ≈ 7B parameters,  D* ≈ 10B tokens                    │
│    D/N ratio ≈ 1.4   ← mostly model-limited!               │
│                                                             │
│  GPT-3 followed this: 175B params × 300B tokens (D/N=1.7)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  CHINCHILLA (2022): "Train smaller models on more data"     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For C = 10²³ FLOPs:                                        │
│    N* ≈ 2.5B parameters,  D* ≈ 50B tokens                  │
│    D/N ratio ≈ 20   ← both N and D scale equally!          │
│                                                             │
│  Chinchilla 70B × 1.4T tokens: D/N = 20 ✓                 │
│  → MATCHES GPT-3 (175B) on most benchmarks                 │
│     but uses 2.5× fewer parameters (= cheaper inference!)  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Loss Formula

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   L(N, D) = E  +  A / N^α  +  B / D^β                      │
│                                                              │
│   E = 1.693   (irreducible entropy of natural language)      │
│   A = 406.4,  α = 0.3392   (model capacity term)            │
│   B = 410.7,  β = 0.2849   (data coverage term)             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 Pipeline Summary

```
GIVEN: compute budget C (FLOPs)
        ↓
Step 1: Use C = 6 × N × D  (FLOP formula for dense Transformer)
        ↓
Step 2: Minimise L(N, D) subject to C = 6ND
        (Lagrange multiplier optimisation)
        ↓
Step 3: Find N* ≈ 0.12 × C^{0.49},  D* ≈ C^{0.51} / 0.72
        ↓
Step 4: D*/N* ≈ 20 (the Chinchilla ratio)
        ↓
OUTPUT: optimal (N*, D*) for your compute budget
```

---

## 2. The Parametric Loss Model

### 2.1 Formula and Constants

```
L(N, D) = E  +  A / N^α  +  B / D^β

TERM INTERPRETATIONS:

  E = 1.693 nats/token
    The "floor" — the minimum possible loss regardless of N and D.
    This is the irreducible entropy of the natural language distribution.
    No model can do better than this (without memorisation).

  A / N^α:
    Contribution from model capacity.
    As N → ∞: this term → 0 (infinite parameters → perfect representation)
    α = 0.34: doubling N reduces this term by 2^{0.34} = 1.27×

  B / D^β:
    Contribution from data coverage.
    As D → ∞: this term → 0 (infinite data → perfect statistics)
    β = 0.28: doubling D reduces this term by 2^{0.28} = 1.21×
```

### 2.2 Numerical Examples

```
EXAMPLE 1: Chinchilla 70B model (N=70B, D=1.4T tokens)

  N^α = (70×10⁹)^{0.3392}
      = 10^{10×0.3392} × 70^{0.3392}
      = 10^{3.392} × 3.66
      = 2466 × 3.66 = 9025

  A/N^α = 406.4 / 9025 = 0.045

  D^β = (1.4×10¹²)^{0.2849}
      = 10^{12×0.2849} × 1.4^{0.2849}
      = 10^{3.419} × 1.096
      = 2625 × 1.096 = 2877

  B/D^β = 410.7 / 2877 = 0.143

  L = 1.693 + 0.045 + 0.143 = 1.881 nats  → PPL = exp(1.881) = 6.56 ✓
  (Chinchilla 70B actual PPL ≈ 6.7 on standard benchmarks — very close!)

EXAMPLE 2: GPT-3 (N=175B, D=300B)

  N^α ≈ 14120 → A/N^α ≈ 0.029
  D^β ≈ 2129  → B/D^β ≈ 0.193  ← much higher! (data-limited)

  L ≈ 1.693 + 0.029 + 0.193 = 1.915 nats  → PPL ≈ 6.79

  GPT-3 has lower "parameter loss" but higher "data loss" — it's data-starved.
  Chinchilla 70B (3.5× smaller) beats GPT-3 by being properly data-fed!
```

---

## 3. Deriving Optimal N and D

### 3.1 Lagrange Multiplier Derivation

```
┌─────────────────────────────────────────────────────────────┐
│  OPTIMISATION PROBLEM:                                      │
│                                                             │
│  min_{N,D}  L(N,D) = E + A/N^α + B/D^β                    │
│  subject to:  6ND = C  (fixed compute budget)               │
└─────────────────────────────────────────────────────────────┘

STEP 1: Substitute constraint D = C/(6N) into L:

  L(N) = E + A×N^{-α} + B×(6N/C)^β
        = E + A×N^{-α} + B×(6/C)^β × N^β

STEP 2: Take derivative and set to zero:

  dL/dN = -α×A×N^{-(α+1)} + β×B×(6/C)^β × N^{β-1} = 0

STEP 3: Solve for N:

  β×B×(6/C)^β × N^{β-1} = α×A × N^{-(α+1)}
  β×B×(6/C)^β = α×A × N^{-(α+β)}
  N^{α+β} = α×A / (β×B×(6/C)^β)
           = α×A×C^β / (β×B×6^β)

  N* = [α×A×C^β / (β×B×6^β)]^{1/(α+β)}

STEP 4: Plug in Chinchilla constants:

  α+β = 0.3392 + 0.2849 = 0.6241
  
  α×A = 0.3392 × 406.4 = 137.8
  β×B = 0.2849 × 410.7 = 116.9
  
  Ratio: α×A / (β×B) = 1.179
  
  N* ≈ 1.179^{1/0.6241} × C^{β/(α+β)} / 6^{β/(α+β)}
     ≈ 0.121 × C^{0.4564}

  D* = C / (6 × N*) ≈ C^{0.5436} / 0.726
```

### 3.2 The D/N ≈ 20 Rule

```
┌─────────────────────────────────────────────────────────────┐
│  MARGINAL BENEFIT EQUALITY AT OPTIMUM:                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  At the optimal (N*, D*), the marginal improvement from    │
│  adding one more parameter EQUALS the marginal improvement  │
│  from adding one more training token:                       │
│                                                             │
│  α×A / N^{α+1} = β×B / D^{β+1}                            │
│                                                             │
│  This is the optimality condition (∂L/∂N = ∂L/∂D at C=6ND)│
│                                                             │
│  Solving numerically with fitted constants:                 │
│  D*/N* ≈ 20                                                 │
│                                                             │
│  DECISION TABLE:                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  D/N < 20: model-limited (add more data > add params)│  │
│  │  D/N = 20: optimal balance ✓                         │  │
│  │  D/N > 20: data-limited (add params > add more data) │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  MODEL EXAMPLES:                                            │
│  GPT-3 175B, 300B tokens: D/N = 1.7   ← severely model-ltd │
│  Chinchilla 70B, 1.4T:    D/N = 20    ← Chinchilla-optimal  │
│  LLaMA-3 8B, 15T:         D/N = 1875  ← intentionally over! │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. The Overtrained Model Strategy

### 4.1 Training Cost vs Inference Cost

```
TOTAL DEPLOYMENT COST for a model serving Q queries:

  Total Cost = C_train + C_infer × Q

  C_train = 6 × N × D   (training FLOPs)
  C_infer = 2 × N       (FLOPs per token at batch=1)

ANALYSIS for Q = 1,000,000,000 queries (1B):

  Option A: Chinchilla-optimal 70B (D=1.4T):
    C_train = 6 × 70B × 1.4T = 5.88 × 10²³ FLOPs
    C_infer per query = 2 × 70B = 140B FLOPs
    Total inference: 140B × 10⁹ = 1.4 × 10²⁰ FLOPs

  Option B: Overtrained 7B (D=15T, same compute):
    C_train = 6 × 7B × 15T ≈ 6.3 × 10²³ FLOPs  (slightly more)
    C_infer per query = 2 × 7B = 14B FLOPs
    Total inference: 14B × 10⁹ = 1.4 × 10¹⁹ FLOPs

  ┌────────────────────────────────────────────────────────────┐
  │  Inference cost ratio: 1.4×10²⁰ / 1.4×10¹⁹ = 10× !       │
  │                                                            │
  │  If overtrained 7B matches Chinchilla 70B quality:         │
  │  → 10× cheaper per query                                   │
  │  → For 1B queries: saves 1.26×10²⁰ FLOPs ≈ 100 GPU-years! │
  └────────────────────────────────────────────────────────────┘
```

### 4.2 LLaMA's Approach

```
┌─────────────────────────────────────────────────────────────┐
│  LLAMA-3 STRATEGY: Deliberate Over-training                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model      N (params)   D (tokens)   D/N    Strategy       │
│  ─────────────────────────────────────────────────────────  │
│  LLaMA-3 8B    8B          15T        1875   Over-trained    │
│  LLaMA-3 70B   70B         15T        214    Over-trained    │
│  LLaMA-3 405B  405B        15T        37     Near-Chinchilla │
│                                                             │
│  For 8B: Chinchilla-optimal would be D ≈ 20 × 8B = 160B    │
│  LLaMA uses D = 15T = 94× MORE than Chinchilla-optimal!    │
│                                                             │
│  WHY?  LLaMA-3 8B is designed for INFERENCE efficiency.    │
│  Meta expects billions of API calls → cheap inference > opt training │
│                                                             │
│  EMPIRICAL RESULT:                                          │
│  LLaMA-3 8B (15T tokens) OUTPERFORMS Llama-1 65B (1.4T)   │
│  on most benchmarks, with 8× lower inference compute!      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Isoloss Contours

```
ISOLOSS CONTOUR: the set of (N,D) pairs achieving the same loss L₀.

A×N^{-α} + B×D^{-β} = L₀ − E   (a constant Δ)

These are convex curves in log(N) vs log(D) space:

log D ↑
      │                     ╌╌╌╌╌ L = 1.75 (excellent)
      │                 ╌╌╌╌╌
15T ──┼─────────────────●  ╌╌╌╌╌ L = 1.85
      │             ╌╌╌╌╌
1T  ──┼─────────────●    ╌╌╌╌╌   L = 1.95
      │        ╌╌╌╌╌
100B ─┼────────●                  L = 2.05
      │
      └──────────────────────────── log N
             1B   10B  100B  1T

●  marks the Chinchilla-optimal point for each compute budget

OPTIMAL FRONTIER:
  Each point ● satisfies: D*/N* = 20 and C = 6×N×D

  C = 10²¹:  N* ≈ 0.5B,   D* ≈ 10B
  C = 10²²:  N* ≈ 2B,     D* ≈ 40B
  C = 10²³:  N* ≈ 8B,     D* ≈ 160B
  C = 10²⁴:  N* ≈ 30B,    D* ≈ 600B
  C = 10²⁵:  N* ≈ 110B,   D* ≈ 2.2T
```

---

## 6. Summary

### 6.1 Formulas Quick Reference

**Chinchilla Loss Model:**

```
L(N, D) = E + A/N^α + B/D^β
  E=1.693, A=406.4, α=0.3392, B=410.7, β=0.2849
```

**Compute Formula:**

```
C ≈ 6 × N × D   (FLOPs for dense Transformer training)
```

**Optimal Allocation:**

```
N* ≈ 0.12 × C^{0.48}
D* ≈ C^{0.52} / 0.73
D*/N* ≈ 20
```

### 6.2 Kaplan vs Chinchilla

| Aspect | Kaplan (2020) | Chinchilla (2022) |
|--------|---------------|-------------------|
| Optimal D/N | ~1.5 | ~20 |
| N* scaling | ∝ C^{0.73} | ∝ C^{0.48} |
| D* scaling | ∝ C^{0.27} | ∝ C^{0.52} |
| Verdict | Models undertrained | Balanced training |
| Example error | GPT-3: N=175B, D=300B | Should have been ~10B, ~200B |

### 6.3 Common Mistakes

```
❌ WRONG: Chinchilla-optimal = best model to deploy
✓ RIGHT:  Chinchilla-optimal = lowest loss for GIVEN training budget.
          For DEPLOYMENT, inference cost matters too.
          Smaller overtrained models (like LLaMA-3 8B) are cheaper to serve.

❌ WRONG: D/N = 20 applies to all compute budgets
✓ RIGHT:  D/N ≈ 20 is approximately constant across budgets, but it comes
          from the specific fitted constants α, β. With different data quality
          or architecture, the ratio may differ.

❌ WRONG: C = N × D (forgetting the factor of 6)
✓ RIGHT:  C ≈ 6ND for dense Transformers (forward + backward ≈ 6 FLOPs per param per token).
          For inference-only (forward pass): C ≈ 2ND.

❌ WRONG: The E = 1.693 term means no model can reach PPL < e^{1.693} ≈ 5.4
✓ RIGHT:  E is the irreducible entropy for the specific training DATA used.
          With higher quality data, E may be lower.
          With memorisation of test data, PPL can be arbitrarily low (cheating).
```

---

## 7. Exercises

1. **Loss Calculation**: Using L(N,D) = E + A/N^α + B/D^β, compute L for N=7B and D=1T (LLaMA-1 7B settings). What PPL does this correspond to? Compare to LLaMA-3 8B (D=15T).

2. **Optimal Allocation**: For C = 6 × 10²³ FLOPs, compute N* and D* using the Chinchilla formula. What is D*/N*? How does it compare to 20?

3. **Overtrained vs Optimal**: A company has C = 10²³ training FLOPs and expects Q = 5 × 10⁸ inference queries. Should they train the Chinchilla-optimal model or an overtrained smaller model? Compute total (training + inference) FLOPs for both options.

4. **Marginal Equality**: Show algebraically that at the Chinchilla optimum, ∂L/∂N × N = ∂L/∂D × D (the elasticities are equal). Interpret this in plain language.
