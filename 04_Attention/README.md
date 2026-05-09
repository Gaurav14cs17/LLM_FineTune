# Chapter 04 — Attention Mechanisms

Attention reweights values based on **compatibility** between queries and keys: the scaled dot-product map is the workhorse of Transformers. This chapter derives attention from kernel-style similarity, extends to **multi-head** subspaces, introduces **GQA/MQA** for KV-cache efficiency, specifies **causal** masking for autoregressive decoders, and covers **FlashAttention**’s IO-aware exact implementation.

```
  For each position i (query) attend over prior keys j ≤ i (decoder)

       scores_ij  =  (q_i · k_j) / √d_h      +   optional bias/mask
                        │
                        ▼
                 softmax along j  (row-wise)
                        │
                        ▼
                 sum_j  α_ij  v_j     ∈ ℝ^{d_h}

  Multi-head: h parallel attention heads → concat → W_o
```

## Sub-chapters

**`01_Scaled_Dot_Product/`**  
Shows \(1/\sqrt{d_h}\) scaling stabilises softmax input magnitude as width grows; derives gradients and notes **temperature** analogy when scaling changes.

**`02_Multi_Head_Attention/`**  
Splits \(d\) into \(h\) heads of size \(d_h=d/h\); cost of \(Q,K,V\) and output projections; explains information **factorisation** and head specialisation observed empirically.

**`03_Grouped_Query_Attention/`**  
**MQA** shares one \(K,V\) across many Q heads; **GQA** groups heads to trade accuracy for KV memory. Derives cache-size scaling \(\propto G/h\) relative to full MHA.

**`04_Causal_Masking/`**  
Implements autoregressive training by setting illegal \((i,j)\) pairs to \(-\infty\) before softmax; connects to efficient **triangular** kernel fusion.

**`05_FlashAttention/`**  
Reorders attention into **tiling** that keeps SRAM resident blocks; memory \(\Theta(N)\) for exact attention versus \(\Theta(N^2)\) materialised matrices.

## Key formulas and concepts

**Scaled attention**

\[
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\Bigl(\frac{QK^\top}{\sqrt{d_h}}\Bigr)V.
\]

**Multi-head**

\[
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O,\quad \mathrm{head}_i=\mathrm{Attn}(XW_Q^i,XW_K^i,XW_V^i).
\]

**Causal mask**

\[
M_{ij}=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
\]

**Complexity (dense)**  
\(\Theta(N^2 d)\) time for sequence length \(N\) and projection width on the order of \(d\); memory for materialised attention matrix \(\Theta(N^2)\) without Flash-style kernels.

**Head abstraction**  
Group queries so each **logical** head still attends, but KV tensors repeat: parameter savings are mostly in **inference**, less in training FLOPs unless kernels fuse the sharing aggressively.

## Prerequisites

- Softmax and row-stochastic matrices; masks in floating-point precision.
- Mini-batch tensor layout intuition (`[batch, heads, seq, dim]`).

## Operational cheatsheet

| Concern | Mechanism |
|---------|-----------|
| Long contexts cost | Quadratic attention; sparsity / linear variants (outside this folder) |
| Inference RAM | GQA/MQA shrink KV; Chapter 08 quantises/cache |
| Numerics | FP16 softmax; Flash fuses scale + softmax |

Attention weight heatmaps are **one** decomposition of computation—not a faithful causal explanation of "what the model thinks"; interpret with care.

## When attention dominates wall-clock

- **Prefill** (prompt processing): often compute-bound; FlashAttention matters.
- **Decode** (one new token with KV cache): memory-bandwidth on KV + small matmuls; GQA changes the story (Chapter 08).

## Mini glossary

| Term | Meaning |
|------|---------|
| Logit | Pre-softmax score for a destination position / head |
| KV cache | Stored \(k,v\) **per layer** for past tokens at decode time (Ch. 08) |
| GQA | Grouped-query: subset of heads share KV (`num_kv_heads < num_heads`) |

## Numerical stability note

- Use fused softmax where available; \( -\infty \) becomes large negative float (`minftype`) in practice, not literal \(-\infty\).

---

_End of chapter overview._
