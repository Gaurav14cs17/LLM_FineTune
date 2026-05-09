# Chapter 06 — Pre-training

Pre-training is the large-scale self-supervised phase where the model learns from trillions of tokens.
This chapter covers the loss, optimiser, learning rate schedules, and numerical precision.

---

## Sub-topics

| # | Folder | Topic |
|---|--------|-------|
| 1 | `01_Objective_and_Loss/` | NLL loss, causal LM objective |
| 2 | `02_AdamW_Optimizer/` | Adaptive moments, weight decay, full derivation |
| 3 | `03_LR_Schedules/` | Warmup, cosine decay, WSD |
| 4 | `04_Mixed_Precision/` | BF16/FP16 training, loss scaling |

---

## Key Equations at a Glance

**Training loss:**
```
L(θ) = −(1/T) ∑_{t=1}^{T} log P_θ(wₜ | w₁:ₜ₋₁)
```

**AdamW update:**
```
mₜ = β₁mₜ₋₁ + (1−β₁)gₜ
vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²
m̂ₜ = mₜ/(1−β₁ᵗ),   v̂ₜ = vₜ/(1−β₂ᵗ)
θₜ = θₜ₋₁ − η · m̂ₜ/(√v̂ₜ + ε)  −  η·λ·θₜ₋₁
```

**Cosine LR schedule:**
```
η(t) = η_min + 0.5(η_max − η_min)(1 + cos(πt/T))
```

**FLOPs budget (Chinchilla rule):**
```
C ≈ 6ND     (N = parameters, D = training tokens)
```
