# 04 — Full Decoder Block (LLaMA-3 Architecture)

## 1. Complete Block Diagram

```
                         Input x  (L × d)
                             │
          ┌──────────────────┘
          │  (residual path)
          │                  │
          │            RMSNorm(x)          ← Pre-norm
          │                  │
          │     ┌────────────┼────────────┐
          │     │            │            │
          │    ×W_Q         ×W_K        ×W_V      ← QKV projections
          │     │            │            │
          │  +RoPE         +RoPE          │        ← Rotary PE (only Q,K)
          │     │            │            │
          │     └────────────┼────────────┘
          │                  │
          │     GroupedQueryAttention(Q,K,V)       ← GQA (G=8 groups)
          │     + Causal Mask                      ← lower triangular
          │                  │
          │                ×W_O                    ← output projection
          │                  │
          └────────► + ◄─────┘                    ← residual connection
                     │
                     h  (intermediate, L × d)
                     │
          ┌──────────┘
          │  (second residual path)
          │                  │
          │            RMSNorm(h)          ← Pre-norm
          │                  │
          │     ┌────────────┴───────────┐
          │     │                        │
          │    ×W₁ → SiLU              ×W₃         ← SwiGLU gate & up projections
          │     │                        │
          │     └──────── ⊙ ────────────┘          ← element-wise multiply
          │                  │
          │                ×W₂                      ← down projection
          │                  │
          └────────► + ◄─────┘                    ← residual connection
                     │
                     Output  (L × d)  → next layer
```

## 2. Tensor Shapes Through the Block (LLaMA-3 8B)

```
Configuration: d=4096, h=32 queries, G=8 KV groups, d_k=128, d_ff=14336

Input x:               (B, L, 4096)
RMSNorm(x):            (B, L, 4096)      ← normalised, same shape

QKV projections:
  Q = x @ W_Q:         (B, L, 4096)  →  reshape (B, 32, L, 128)   [32 heads × 128 dim]
  K = x @ W_K:         (B, L, 1024)  →  reshape (B,  8, L, 128)   [8 KV groups × 128 dim]
  V = x @ W_V:         (B, L, 1024)  →  reshape (B,  8, L, 128)

RoPE applied to Q,K:   same shape (rotation is in-place)

GQA:
  Scores: Q @ K.T / √128   (B, 32, L, L) after expanding K  →  causal mask
  Attn weights: softmax(·)  (B, 32, L, L)
  Context: weights @ V      (B, 32, L, 128)
  W_O projection:            (B, L, 4096)

Residual: h = x + attn_out  (B, L, 4096)

RMSNorm(h):            (B, L, 4096)

FFN SwiGLU:
  gate = SiLU(h @ W₁)  (B, L, 14336)
  up   = h @ W₃         (B, L, 14336)
  mid  = gate ⊙ up       (B, L, 14336)
  out  = mid @ W₂        (B, L, 4096)

Output: y = h + ffn_out  (B, L, 4096)  ← same shape as input!
```

## 3. Data Flow Through All 32 Layers

```
Tokens [w₁, w₂, ..., w_L]
         │
         ▼
    Embedding lookup  →  X ∈ ℝ^{L × 4096}
         │
         ▼
    ┌─────────────────────┐
    │  Decoder Block 0    │   (layer 0)
    └─────────────────────┘
         │
         ▼
    ┌─────────────────────┐
    │  Decoder Block 1    │   (layer 1)
    └─────────────────────┘
         │
        ...
         │
         ▼
    ┌─────────────────────┐
    │  Decoder Block 31   │   (layer 31)
    └─────────────────────┘
         │
         ▼
    RMSNorm (final)
         │
         ▼
    lm_head: W_E.T  →  Logits ∈ ℝ^{L × 128000}   (tied with embedding!)
         │
         ▼
    softmax → P(next token) ∈ [0,1]^{128000}
```

## 4. Memory During Training

```
For batch_size=1, seq_len=4096, BF16 (2 bytes):

Component                     Memory
─────────────────────────────────────────
Model parameters (BF16)        8B × 2B = 16 GB
Gradients (BF16)               8B × 2B = 16 GB
AdamW moments (FP32)           8B × 8B = 64 GB   ← 2 states × 4 bytes
Activations (all layers)       ~30 GB (depends on sequence length)
─────────────────────────────────────────
Total                          ~126 GB

Needs 2× A100 80GB minimum for training.
With gradient checkpointing:   activations drop to ~8 GB → fits in ~106 GB.
```

---

## Exercises

1. Sketch the full LLaMA-3 8B block as a data-flow diagram, showing tensor shapes at each step.
2. In GQA with G=8, how many unique K,V pairs are computed per layer per token?
3. Why does weight tying (sharing E and lm_head) not reduce model capacity significantly for generation tasks?
