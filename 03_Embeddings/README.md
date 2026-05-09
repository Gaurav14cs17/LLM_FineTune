# Chapter 03 — Embeddings and Positional Information

Token IDs become **dense vectors** \(e_t \in \mathbb{R}^d\) via lookup tables; positional encodings inject **order** because self-attention (Chapter 04) is **equivariant** to permutations without extra structure. This chapter covers **learned** embeddings, **fixed** sinusoidal bases (original Transformer), **RoPE** (rotary, dominant in open LMs), and **ALiBi** (distance biases in attention logits).

```
  token IDs  t₁ … t_T
       │         │
       ▼         ▼
  E[t_i] ∈ ℝᵈ     +     positional signal π(i) ∈ ℝᵈ   (additive or multiplicative / rotation)
       │                      │
       └──────────┬───────────┘
                  ▼
           x_i  enters attention stacks

  RoPE: encode position by BLOCK ROTATIONS in q,k pairs  (not a separate add vector)
```

## Sub-chapters

**`01_Token_Embeddings/`**  
Defines the matrix \(E \in \mathbb{R}^{|\mathcal{V}|\times d}\), tie-in with output softmax sometimes via **weight tying**, and scaling heuristics (e.g. \(\sqrt{d}\) factors in original Transformers). Discusses **embedding inflation** when \(|\mathcal{V}|\) is large and shard placement in distributed training.

**`02_Sinusoidal_PE/`**  
Derives fixed frequencies \(\omega_i\) so that positions map to unique sin/cos tuples; highlights extrapolation **limitations** to \(T\) beyond training length unless augmented with learned or relative methods.

**`03_RoPE/`**  
Rotary Position Embedding factorises attention into **rotation-invariant** inner products when \(q,k\) pairs are 2D-rotated per frequency bucket; algebraically encodes **relative** offsets without blowing up sequence-length parameters. Includes implementation notes (interleaved vs blocked layouts) that match FlashAttention kernels.

**`04_ALiBi/`**  
Attention with Linear Biases adds a **head-specific linear penalty** \(-m_h|i-j|\) to logits—**no** explicit position embedding tensor; strong for some training regimes though less common than RoPE in recent LLM releases.

## Key formulas and concepts

**Token embedding lookup**

\[
x_i^{\mathrm{(tok)}} = E[t_i].
\]

**Sinusoidal (type-1 PE)**

\[
\begin{aligned}
\mathrm{PE}(t,2i) &= \sin\Bigl(t/10000^{2i/d}\Bigr),\\
\mathrm{PE}(t,2i+1) &= \cos\Bigl(t/10000^{2i/d}\Bigr).
\end{aligned}
\]

**RoPE (schematic 2D block)**

\[
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix}
=
\begin{bmatrix}
\cos(t\theta_i) & -\sin(t\theta_i) \\
\sin(t\theta_i) & \cos(t\theta_i)
\end{bmatrix}
\begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix},
\quad \theta_i = 1/10000^{2i/d}.
\]

**ALiBi bias**

\[
\mathrm{bias}_h(i,j) = -m_h\,|i-j|.
\]

## Prerequisites

- Linear algebra: rotations, inner products, block matrices.
- Chapter 02: how tokens map to rows of \(E\).

## Design axis summary

| Scheme | Params for length | Relative info | Notes |
|--------|--------------------|-----------------|-------|
| Learned absolute | \(T_{\max}\cdot d\) | weak unless large \(T_{\max}\) | simple |
| Sinusoidal | 0 | implicit | extrapolation tricky |
| RoPE | 0 core | strong rotation structure | LLaMA-class default |
| ALiBi | 0 core | strong distance prior | alternate bias story |

## Tensor shape mantra

- **B** batch, **T** time, **H** heads, **D** head dim. RoPE rotates the **last** head-dimension pair; layouts must match the CUDA kernel you call.

## RoPE vs absolute additive PE (design)

- **Additive PE + learned \(E\)**: positions enter as extra vectors; extrapolation requires tricks or very large max length.
- **RoPE**: rotations tie \(q,k\) so **relative phase** carries offset; kernel fusions-friendly; dominant in LLaMA/Gemma/Mistral lines.

## Mini glossary

| Term | Meaning |
|------|---------|
| Weight tying | Same matrix for input embeddings and output logits (optional) |
| Long-context drift | Behaviour past trained \(T_{\max}\) when using relative PE |
| mrope / YaRN | (advanced) rescale frequencies for extrapolation beyond train length |

---

_End of chapter overview._
