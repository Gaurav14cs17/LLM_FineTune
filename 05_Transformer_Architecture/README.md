# Chapter 05 — Transformer Architecture

The full decoder-only Transformer block stacks attention, feed-forward, and normalisation layers.
This chapter covers each component's mathematics and design decisions in modern LLMs.

---

## Sub-topics

| # | Folder | Topic |
|---|--------|-------|
| 1 | `01_Layer_Normalization/` | LayerNorm, RMSNorm, Pre-norm vs Post-norm |
| 2 | `02_Feed_Forward_Network/` | FFN, SwiGLU, parameter count |
| 3 | `03_Residual_Connections/` | Skip connections, gradient flow, depth |
| 4 | `04_Full_Decoder_Block/` | Complete LLaMA-3 block, data flow |

---

## Key Equations at a Glance

**RMSNorm:**
```
RMSNorm(x) = x / RMS(x) · γ     where RMS(x) = √(1/d ∑_i xᵢ²)
```

**SwiGLU FFN:**
```
FFN_SwiGLU(x) = (SiLU(x·W₁) ⊙ (x·W₃)) · W₂
SiLU(x) = x · σ(x) = x / (1 + e^{−x})
```

**Residual block (Pre-norm style, LLaMA):**
```
h = x + Attention( RMSNorm(x) )
y = h + FFN( RMSNorm(h) )
```

**Full decoder block parameter count (d=4096, d_ff=14336):**
```
Attention:   4 × d²         = 67M
FFN (SwiGLU): 3 × d × d_ff  = 176M
Norms:       4d              ≈ 0 (negligible)
Total/layer ≈ 243M
× 32 layers ≈ 7.8B params
```
