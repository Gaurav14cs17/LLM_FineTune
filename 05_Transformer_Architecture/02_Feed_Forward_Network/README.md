# 02 — Feed-Forward Network (FFN) and SwiGLU

## 1. Role of FFN in the Transformer

```
Transformer block:
  Input x
    │
    ├─ Attention ─── RMSNorm ─── residual ──►  h = x + Attn(RMSNorm(x))
    │
    └─ FFN ───────── RMSNorm ─── residual ──►  y = h + FFN(RMSNorm(h))

Attention:  mixes ACROSS tokens (token communication)
FFN:        operates on EACH TOKEN INDEPENDENTLY (token processing)

            "The"  "cat"  "sat"
              ↓      ↓      ↓
           FFN(·) FFN(·) FFN(·)    ← same FFN weights, applied independently
              ↓      ↓      ↓
            out₁   out₂   out₃
```

## 2. Standard FFN (Original Transformer)

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂

Shapes:
  x:       d             (hidden dim)
  W₁:      d → d_ff      (expand to d_ff = 4d)
  ReLU(·): d_ff           
  W₂:      d_ff → d      (project back)

DIMENSION EXPANSION VISUALISED:
  ────────────────────────────────────────────────────────
  Input x:    [─────── d ────────────]    (e.g., d=512)
  W₁·x:       [───────────────────── d_ff ─────────────── ─────────]   (d_ff=2048)
  ReLU:        [         zeros        │  positive values            ]   (sparse!)
  W₂·(·):     [─────── d ────────────]    (back to 512)
  ────────────────────────────────────────────────────────

The expansion to d_ff=4d creates a "memory" that can store many patterns.
Research suggests FFN layers store factual knowledge ("Paris is the capital of France").
```

## 3. SwiGLU — The Modern FFN

```
FFN_SwiGLU(x) = (SiLU(x·W₁)  ⊙  (x·W₃))  ·  W₂
                 ─────────────    ─────────
                   gate signal    content signal

SiLU(x) = x · σ(x) = x / (1 + e⁻ˣ)  (Sigmoid Linear Unit / Swish)

DATA FLOW:
  x (d-dim)
    │
    ├──────► W₁ ──► SiLU(·) ──────────►  gate  ─────┐
    │                                                  ⊙──► W₂ ──► output (d)
    └──────► W₃ ──────────────────────►  content ────┘

The GATE (SiLU output) controls how much of the CONTENT passes through.
This is content-dependent: x itself determines what it lets through.

ACTIVATION SHAPE:
  SiLU(x)  vs  ReLU(x):
  
  ReLU:  0 everywhere < 0, linear for > 0 (hard cutoff)
         y
         │        /
         │       /
         │──────/────── x
         0
  
  SiLU:  small negative values near 0, smooth transition
         y        /
         │       /
         │     _/
         │────/──────── x
         │   (slight dip below 0 near x=-1)
         
  SiLU is smooth everywhere → no "dying ReLU" problem
  SiLU has non-zero gradient everywhere (beneficial for training)
```

## 4. SwiGLU Parameter Sizes

```
Standard FFN (d=512, d_ff=2048):
  W₁: 512 × 2048  = 1,048,576
  W₂: 2048 × 512  = 1,048,576
  Total: 2,097,152  (≈ 2M)

SwiGLU FFN (d=512, d_ff=?):
  W₁, W₂, W₃ each: 512 × d_ff
  Total: 3 × 512 × d_ff
  
  To match 2M params:  3 × 512 × d_ff = 2,097,152
                        d_ff = 1365 = ⌊8d/3⌋

LLaMA-3 8B:  d=4096, d_ff=14336  (slightly larger than 8d/3 = 10922)
  W₁: 4096 × 14336 = 58.7M  }
  W₂: 14336 × 4096 = 58.7M  } Total FFN = 176M per layer
  W₃: 4096 × 14336 = 58.7M  }
```

## 5. Why FFN = Memory?

```
Key observations (Geva et al. 2021 "Transformer Feed-Forward Layers Are Key-Value Memories"):

FFN(x) = W₂ · f(W₁ · x)  can be written as:
  = ∑ᵢ fᵢ(x) · W₂[:, i]   (sum over d_ff "memory" entries)

  W₁[i, :] = key vector   (what input pattern does entry i respond to?)
  W₂[:, i] = value vector  (what information does entry i add to the output?)

Example in GPT:
  W₁[42, :] ≈ "European capital"  (detects this pattern in x)
  W₂[:, 42] ≈ adds "Paris, France" information to output

SPARSE ACTIVATION:
  Only ~2-5% of neurons fire per token (ReLU zeroes most out)
  Each neuron = one "memory slot"
  d_ff = 14336 → 14336 independent memories per layer!
```

---

## Exercises

1. Compute SiLU(0), SiLU(1), SiLU(−1), SiLU(2).
2. For d=512 and SwiGLU with 3 matrices, find d_ff such that FFN has the same params as a standard d_ff=2048 FFN.
3. Explain why the SwiGLU gating is adaptive (input-dependent) while LayerNorm scaling (γ) is static (same for all inputs).
