# Chapter 03 — Embeddings & Positional Encoding

Token embeddings map discrete IDs to continuous vectors. Positional encodings inject sequence order
information since the Transformer's attention is permutation-invariant.

---

## Sub-topics

| # | Folder | Scheme | Used by |
|---|--------|--------|---------|
| 1 | `01_Token_Embeddings/` | Learned lookup table E ∈ ℝ^{|V|×d} | All models |
| 2 | `02_Sinusoidal_PE/` | Fixed sin/cos waves | Original Transformer, GPT-2 |
| 3 | `03_RoPE/` | Rotary Position Embedding | LLaMA, Mistral, Gemma |
| 4 | `04_ALiBi/` | Attention with Linear Biases | MPT, BLOOM |

---

## Key Equations at a Glance

**Token embedding lookup:**
```
x_t = E[token_id_t]   ∈ ℝ^d
```

**Sinusoidal PE:**
```
PE(t, 2i)   = sin(t / 10000^{2i/d})
PE(t, 2i+1) = cos(t / 10000^{2i/d})
```

**RoPE rotation matrix (2×2 block):**
```
R(t,i) = [ cos(t·θᵢ)  −sin(t·θᵢ) ]
          [ sin(t·θᵢ)   cos(t·θᵢ) ]
  where θᵢ = 1 / 10000^{2i/d}
```

**ALiBi bias:**
```
bias(q,k) = −m · |pos_q − pos_k|
  where m is a head-specific slope
```
