# 01 — Kaplan Power Laws

## 1. Empirical Power Laws — Mathematical Framework

```
KAPLAN ET AL. (2020) observed that language model loss follows power laws:

  L(N) = (N_c/N)^{α_N}    (loss as function of parameters N, with data fixed)
  L(D) = (D_c/D)^{α_D}    (loss as function of data D, with params fixed)
  L(C) = (C_c/C)^{α_C}    (loss as function of compute C)

FITTED PARAMETERS (on clean internet text):
  α_N ≈ 0.076   (loss exponent for model size)
  α_D ≈ 0.095   (loss exponent for data)
  α_C ≈ 0.050   (loss exponent for compute)

  N_c ≈ 8.8×10¹³ parameters  (scaling constant for N)
  D_c ≈ 5.4×10¹³ tokens      (scaling constant for D)

WHAT POWER LAWS MEAN:
  Linear plot: L vs N looks like a curve
  Log-log plot: log L vs log N is a STRAIGHT LINE with slope −α_N
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │  log L                                                              │
  │  ↑                                                                  │
  │  │ × ← GPT-2 (117M)                                                │
  │  │       × ← GPT-2 (345M)                                          │
  │  │              × ← GPT-2 (762M)                                   │
  │  │                      × ← GPT-2 (1.5B)                           │
  │  │                             × ← GPT-3 (175B)                    │
  │  │                                                                  │
  │  └───────────────────────────────────────────────────────── log N   │
  │           slope ≈ −0.076 (Kaplan)                                   │
  └─────────────────────────────────────────────────────────────────────┘
  
  STRAIGHT LINE IN LOG-LOG MEANS: every 10× increase in N → loss drops by 10^{0.076} ≈ 1.19×.
  (About 19% loss reduction per decade in parameter count)
```

## 2. The Joint Scaling Law — Derivation

```
KAPLAN combined all three axes (N, D, C) into one formula:

  L(N, D) = [(N_c/N)^{α_N/α_D}  +  D_c/D]^{α_D}

  with C ≈ 6·N·D (FLOPs = 6 × params × tokens, for dense Transformer)

DERIVATION from individual power laws:
  L_N(N) = (N_c/N)^{α_N}   at D → ∞
  L_D(D) = (D_c/D)^{α_D}   at N → ∞
  
  The combined form assumes the two contributions ADD (in the appropriate metric space).
  Taking L^{1/α_D} as the "effective difficulty" gives:
  
  L^{1/α_D} = (N_c/N)^{α_N/α_D} + (D_c/D)
  
  Both terms must be equal at the optimal allocation:
    (N_c/N*)^{α_N/α_D} = D_c/D*
    → N* ∝ D*^{α_D/α_N} = D*^{0.095/0.076} = D*^{1.25}

KAPLAN OPTIMAL ALLOCATION:
  At fixed C = 6ND:
    If N is too large (data-limited): L dominated by (D_c/D)^{α_D}
    If N is too small (model-limited): L dominated by (N_c/N)^{α_N}
    
  Kaplan recommendation: N* ∝ C^{0.73}  (allocate most compute to larger models)
  D* ∝ C^{0.27}  (keep data relatively small — LESS aggressive data scaling)
  
  Kaplan's D*/N* ratio ≈ C^{0.27}/C^{0.73} ∝ C^{-0.46} → D/N ratio DECREASES with C!
  → "Larger compute budgets → train bigger models on relatively less data"
  → GPT-3 followed this: 175B params, 300B tokens (ratio ≈ 1.7×)
  
  THIS WAS LATER SHOWN TO BE WRONG by Chinchilla (see next chapter).
```

## 3. The Power Law Exponent — What α Means

```
MATHEMATICAL INTERPRETATION of exponent α:

  L(N) = (N_c/N)^α
  ∂L/∂N = −α × N_c^α × N^{-(α+1)} = −α × L / N

  "Marginal value of adding one more parameter":
    ΔL ≈ −(α × L / N) × ΔN
    
    RELATIVE IMPROVEMENT: ΔL/L ≈ −α × ΔN/N
    
    Doubling N (ΔN/N = 1): ΔL/L ≈ −α → loss decreases by fraction α
    More precisely: L(2N)/L(N) = (N_c/2N)^α / (N_c/N)^α = (1/2)^α = 2^{-α}
    
    For α=0.076: L(2N) = L(N) × 2^{-0.076} = L(N) × 0.949
    → Doubling params reduces loss by 5.1%

COMPARE α VALUES:
  α_N ≈ 0.076 (params): doubling params → 5.1% loss reduction
  α_D ≈ 0.095 (data):   doubling data  → 6.3% loss reduction
  α_C ≈ 0.050 (compute): doubling C    → 3.4% loss reduction
  
  → Data scaling is MORE efficient per-doubling than parameter scaling!
  → Chinchilla confirms this: D and N should scale equally.

CONVERGENCE RATE ANALYSIS:
  How many doublings of N to halve the loss?
    L(N × 2^k) = L(N) × 2^{-α_N × k}
    Set L(N × 2^k) = L(N)/2: 2^{-α_N × k} = 0.5 → k = 1/α_N ≈ 13.2
    
    → Need 13.2 DOUBLINGS (10000× more parameters) to halve the loss!
    (This shows diminishing returns — language is hard to compress)
```

## 4. Early Stopping and Data Budget

```
KAPLAN CRITICAL INSIGHT on training tokens:
  Training longer on the same data has diminishing returns.
  
  For N-parameter model, there's an OPTIMAL number of training tokens D*(N):
    D*(N) ≈ N × (N_c/D_c)^{α_N/α_D}
           ≈ N × (8.8e13/5.4e13)^{0.076/0.095}
           ≈ N × 1.63^{0.8}
           ≈ 1.5 × N  tokens

  KAPLAN: Train each model on ~1.5× its parameter count in tokens.
  
  GPT-3 (175B params) → Kaplan-optimal: ~262B tokens
  Actual: trained on 300B tokens (slightly more, reasonable)

WHY MORE TOKENS EVENTUALLY STOPS HELPING (Kaplan):
  Memorisation: model starts to overfit rare patterns in the training corpus.
  Diversity: corpus is finite → same documents appear multiple times → reduced signal.
  
  OVERFITTING METRIC:
    Train loss ↘ (always)
    Val loss ↘ → then plateau (optimal stopping point)
    
    ┌────────────────────────────────────────────────────────────┐
    │  L                                                         │
    │  ↑                                                         │
    │  │ train loss \────────────────────────                   │
    │  │             \                                           │
    │  │  val loss    \──────────────────\────────── (plateau)  │
    │  │               \                 ↑                      │
    │  │                \─────────       D*(N)  optimal         │
    │  └──────────────────────────────────────────────── D (tokens)│
    └────────────────────────────────────────────────────────────┘
```

## 5. Compute-Optimal Frontier — Kaplan vs Chinchilla

```
KAPLAN (2020) CONCLUSION:
  For a fixed compute budget C = 6ND:
    Optimal: N* ∝ C^{0.73},  D* ∝ C^{0.27}
    D*/N* ratio ≈ 1.5 (small relative to Chinchilla's 20!)
    
  EFFECT: Large models trained on relatively little data.
    GPT-3: 175B params × 300B tokens → D/N = 1.7
    
CHINCHILLA (2022) REVISION:
  For a fixed compute budget C = 6ND:
    Optimal: N* ∝ C^{0.49},  D* ∝ C^{0.51}
    D*/N* ratio ≈ 20
  
  EFFECT: Medium models trained on much more data.
    Chinchilla 70B × 1.4T tokens → D/N = 20 ✓

WHY THE DISCREPANCY?
  Kaplan used training runs that were TOO SHORT (underfitted).
  Their D* estimates were based on early-stopped models.
  If models were trained until convergence, the optimal D/N would increase.
  
  Kaplan's experiment: train each N to near-convergence ON 300B TOKENS.
    → Some models are data-limited (N large), biasing the results.
    Chinchilla varied BOTH N and D simultaneously → more comprehensive.

SUMMARY TABLE:
  C (FLOPs)  │ Kaplan N*    │ Kaplan D*   │ Chinchilla N*│ Chinchilla D*
  ───────────┼──────────────┼─────────────┼──────────────┼──────────────
  10²²       │  ~1.3B       │  ~1.9B tok  │  ~0.5B       │  ~11B tok
  10²³       │  ~7B         │  ~10B tok   │  ~2B         │  ~40B tok
  10²⁴       │  ~36B        │  ~54B tok   │  ~9B         │  ~170B tok
  10²⁵       │  ~200B       │  ~300B tok  │  ~40B        │  ~800B tok
```

---

## Exercises

1. Using L(N) = (N_c/N)^{0.076} with N_c = 8.8×10¹³, compute L for N=7B and N=70B. What is the loss ratio? How many times do you need to double N from 7B to reach 70B's performance?
2. For a fixed budget C=6×10²³ FLOPs, use both Kaplan and Chinchilla rules to compute N* and D*. Show the D/N ratio for each.
3. Prove that if L ∝ N^{-α} and L ∝ D^{-β} with α < β, then data scaling is more "efficient" per doubling than model scaling. Compute the ratio of loss improvements per doubling.
