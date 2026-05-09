# Chapter 05 — Transformer Architecture

This chapter composes attention (Chapter 04), embeddings (Chapter 03), and feed-forward blocks into a **decoder-only** stack as used in GPT and LLaMA families. You learn normalisation placement (**pre-norm** vs **post-norm**), SwiGLU vs classic MLPs, residual pathways for gradient flow, and how to **account parameters** for a realistic training estimate.

```
  x_ℓ  ──►  RMSNorm ──►  Attention ──►  + residual ──►  RMSNorm ──►  FFN(SwiGLU) ──►  +  ──►  x_{ℓ+1}
            (optional dropout / MoE branch not shown)

  Depth L, width d, FFN width d_ff  ⇒  parameter budget scales ~ L * (d² + d d_ff)
```

## Sub-chapters

**`01_Layer_Normalization/`** and **`01_RMSNorm/`**  
LayerNorm centres and rescales activations across the hidden dimension; **RMSNorm** drops centering and only normalises the root-mean-square, saving compute and often matching pretraining stability in large LMs. Explains **affine** \(\gamma\) / optional bias and where they attach in **pre-norm** stacks.

**`02_Feed_Forward_Network/`** and **`02_FFN_and_SwiGLU/`**  
The two-matrix ReLU/GELU MLP versus **SwiGLU** with three projections \(W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}\). Derives the **\(8d/3\)** inner-width rule to preserve parameter/FLOP parity versus a classical **\(4d\)** expansion; discusses deliberate over-width (**\(3.5d\)**) in LLaMA-3-class models.

**`03_Residual_Connections/`**  
Residuals \(x \leftarrow x + f(x)\) stabilise depth \(L\); discusses gradient super-highways, **Post-Deep** scaling tricks (brief), and why pre-norm allows larger \(L\) at fixed precision.

**`04_Full_Decoder_Block/`**  
End-to-end tensor story from token IDs through final logits; ties RMSNorm ordering (LLaMA) vs LayerNorm + learned PE (GPT-2) contrasts; parameter/FLOP tallies per layer and whole model.

## Key formulas and concepts

**RMSNorm**

\[
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma.
\]

**Pre-norm decoder block (schematic)**

\[
\begin{aligned}
h &= x + \mathrm{Attn}(\mathrm{Norm}(x)),\\
y &= h + \mathrm{FFN}(\mathrm{Norm}(h)).
\end{aligned}
\]

**SwiGLU**

\[
\mathrm{FFN}(x) = \bigl(\mathrm{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\bigr) W_{\text{down}}.
\]

**Rough per-layer parameter intuition (dense)**  
Attention \( \sim 4 d^2 \) (with separate \(W_Q,W_K,W_V,W_O\)); FFN \(\sim 3 d d_{\text{ff}}\) for SwiGLU—often **dominant** when \(d_{\text{ff}} \gg d\).

## Prerequisites

- Chapters 03–04 (positional info + attention).
- Basic matrix counting for parameter estimates.

## Chapter placement

| Needs from earlier | Feeds into |
|--------------------|------------|
| RoPE, embeddings | Full block diagrams |
| Multi-head + GQA | Layer layout with KV head repeats |
| Scaling laws (Ch. 09) | Translate \(N\) into concrete \(L,d,d_{\text{ff}}\) choices |

## Duplicate folders note

This repo keeps parallel paths (`01_Layer_Normalization` vs `01_RMSNorm`, `02_Feed_Forward_Network` vs `02_FFN_and_SwiGLU`) to support alternative lesson orders—content overlaps intentionally; pick one track per pass.

## Study tips

- Recompute parameter totals for a published config (e.g. \(d=4096, L=32\)) and compare to reported `total_params`.
- Sketch backward-pass memory footprint difference between **GELU-MLP** and **SwiGLU** at matched budget vs matched inner width.

## Parameter accounting template (dense decoder)

| Block | Schematic count | Dominant when |
|-------|-----------------|---------------|
| Embeddings | \(\|\mathcal{V}\| d\) | huge vocab / tied logits |
| Per-layer self-attn | \(\sim 4d^2\) (four projections, square-ish) | MoE off |
| Per-layer FFN SwiGLU | \(3 d d_{\text{ff}}\) | \(d_{\text{ff}} \sim 3\)–\(4\times d\) |
| LM head (if untied) | \(\|\mathcal{V}\| d\) again | often tied away |

## Stability culture

- **Pre-norm + RMSNorm** is the default in many open LLMs: expect LayerNorm algebra in papers but RMS in code.
- Watch **width multipliers of 64/128** for TP/DP shard alignment—they slightly perturb \(d_{\text{ff}}\) vs ideal \(8d/3\).

## Depth vs width (rule-of-thumb discussion)

- Increasing \(L\) deepens **multi-hop computation paths** through residual super-highways; increasing \(d\) widens **representational bandwidth**. Scaling-law studies often move both together—do not infer causality from public configs alone.

## Intended outcomes

After this chapter you should: (1) expand a stacked LM config into **parameter tables**; (2) explain why SwiGLU widens FFNs differently from a classic \(4d\) MLP; (3) diagram **pre-norm** residual flow versus **post-norm**; (4) estimate the relative cost of attention vs FFN FLOPs given \(d,d_{\text{ff}},L\).

## Exam-style checks

- Given \(d\) and \(d_{\text{ff}}\), compute SwiGLU FFN parameters as \(3dd_{\text{ff}}\) and compare to \(2d\cdot 4d\) for a classical MLP—state which is larger when \(d_{\text{ff}}=3.5d\).
- Explain why RMSNorm subtracts **no mean** compared to LayerNorm when training stability is discussed in slides versus code.

---

_End of chapter overview._
