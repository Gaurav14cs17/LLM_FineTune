# Chapter 07 — Fine-Tuning

Fine-tuning adapts a pre-trained LLM to follow instructions, align with human preferences, or
specialise for a domain. This chapter covers the mathematical foundations of all major methods.

---

## Sub-topics

| # | Folder | Method | Core Idea |
|---|--------|--------|-----------|
| 1 | `01_SFT/` | Supervised Fine-Tuning | NLL on demonstration data |
| 2 | `02_LoRA/` | Low-Rank Adaptation | W = W₀ + AB, freeze W₀ |
| 3 | `03_QLoRA/` | Quantized LoRA | NF4-quantized W₀ + BF16 LoRA |
| 4 | `04_RLHF_PPO/` | RLHF with PPO | Reward model + PPO policy gradient |
| 5 | `05_DPO/` | Direct Preference Optimization | Closed-form policy from preferences |

---

## Key Equations at a Glance

**SFT loss:**
```
L_SFT(θ) = −∑_t log P_θ(yₜ | x, y<t)   (supervised tokens only)
```

**LoRA reparameterisation:**
```
W = W₀ + ΔW = W₀ + B·A    where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d,k)
```

**LoRA parameter savings:**
```
Full fine-tune: d×k   LoRA: r×(d+k)   Reduction: dk / r(d+k) ≈ d/r
```

**DPO loss:**
```
L_DPO(θ) = −E_{(x,yw,yl)~D} [ log σ( β·log[π_θ(yw|x)/π_ref(yw|x)] − β·log[π_θ(yl|x)/π_ref(yl|x)] ) ]
```

**PPO clipped objective:**
```
L_CLIP(θ) = −E_t [ min( rₜ(θ)·Âₜ,  clip(rₜ(θ), 1−ε, 1+ε)·Âₜ ) ]
rₜ(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)
```
