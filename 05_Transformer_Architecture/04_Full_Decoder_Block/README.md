# The Complete Decoder Block (LLaMA-3 Style)

*Pre-norm, GQA, SwiGLU, residuals, FLOPs, parameters, and the embedding-to-logits pipeline*

---

## Table of Contents

1. [Overview](#1-overview)
2. [INTUITION: One Block as Two Residuals](#2-intuition-one-block-as-two-residuals)
3. [Forward Pass: RMSNorm → MHA → Residual → RMSNorm → FFN → Residual](#3-forward-pass-rmsnorm--mha--residual--rmsnorm--ffn--residual)
4. [VARIABLES (LLaMA-3 8B Reference)](#4-variables-llama-3-8b-reference)
5. [Layer Shapes and GQA](#5-layer-shapes-and-gqa)
6. [FLOPs per Decoder Block](#6-flops-per-decoder-block)
7. [Parameter Count per Layer](#7-parameter-count-per-layer)
8. [Memory per Layer (Training Sketch)](#8-memory-per-layer-training-sketch)
9. [Stacking 32 Blocks and the Full Pipeline](#9-stacking-32-blocks-and-the-full-pipeline)
10. [NUMERICAL WALKTHROUGH](#10-numerical-walkthrough)
11. [Pseudocode](#11-pseudocode)
12. [COMMON MISTAKES](#12-common-mistakes)
13. [EXERCISES](#13-exercises)

---

## 1. Overview

A **decoder-only Transformer block** (e.g. LLaMA-3) repeats two **pre-norm residual** sublayers:

1. **Self-attention** (here **Grouped-Query Attention**, GQA): \(\mathbf{x} \mapsto \mathbf{x} + \mathrm{MHA}(\mathrm{RMSNorm}(\mathbf{x}))\).
2. **Feed-forward** (**SwiGLU** MLP): \(\mathbf{h} \mapsto \mathbf{h} + \mathrm{FFN}(\mathrm{RMSNorm}(\mathbf{h}))\).

No encoder: a single stream of token vectors \(\mathbf{X}\in\mathbb{R}^{L\times d}\) flows through \(N\) blocks, then **final RMSNorm** and **linear head** to vocabulary logits. This note fixes **FLOP** and **parameter** accounting for one block, shows how **32** such blocks compose, and gives a **numeric** trace.

---

## 2. INTUITION: One Block as Two Residuals

```
ONE DECODER LAYER  (repeated N times, e.g. N=32)

     x  ─────┬────────────────────────────────────────►  +  ──►  h
             │                                            ▲
             │    RMSNorm                                 │
             │       │                                    │
             │       ▼                                    │
             │    ┌──────────────────┐                    │
             │    │ MHA + RoPE + mask│                    │
             │    └────────┬─────────┘                    │
             │             │                            │
             │          ×W_O ────────────────────────────┘
             │
             └── skip path (identity on raw x)

     h  ─────┬────────────────────────────────────────►  +  ──►  y  (next layer input)
             │                                            ▲
             │    RMSNorm                                 │
             │       │                                    │
             │       ▼                                    │
             │   gate = SiLU(h W₁)   up = h W₃           │
             │        └────── ⊙ ──────┘                   │
             │               │                            │
             │            × W₂ ───────────────────────────┘
             │
             └── skip path (identity on raw h)

OUTPUT y has SAME shape as x: (batch, seq_len, d_model).
```

---

## 3. Forward Pass: RMSNorm → MHA → Residual → RMSNorm → FFN → Residual

**Notation:** batch \(B\), sequence \(L\), model width \(d\), FF inner \(d_{\mathrm{ff}}\), heads \(H\), KV groups \(G\), head dim \(d_h = d/H\).

1. **Input** \(\mathbf{x} \in \mathbb{R}^{B\times L\times d}\).
2. \(\hat{\mathbf{x}} = \mathrm{RMSNorm}(\mathbf{x})\) (per-row RMS over \(d\), optional learnable scale \(\boldsymbol{\gamma}\)).
3. **Project** \(\mathbf{Q} = \hat{\mathbf{x}}\mathbf{W}_Q\), \(\mathbf{K} = \hat{\mathbf{x}}\mathbf{W}_K\), \(\mathbf{V} = \hat{\mathbf{x}}\mathbf{W}_V\) with **GQA**: \(\mathbf{W}_K,\mathbf{W}_V\) map to \(G\times d_h\) **per head group** (not full \(H\times d_h\) unless \(G=H\)).
4. **RoPE** applied to \(\mathbf{Q},\mathbf{K}\) (positions \(0..L{-}1\)).
5. Scores \(\mathbf{S} = \mathbf{Q}\mathbf{K}^{\mathsf T}/\sqrt{d_h}\); **causal mask** \(-\infty\) above diagonal; **softmax** \(\to \mathbf{A}\); **context** \(\mathbf{A}\mathbf{V}\); merge heads; **output** \(\mathbf{O}=\mathrm{merge}(\mathbf{A}\mathbf{V})\mathbf{W}_O\).
6. **Residual:** \(\mathbf{h} = \mathbf{x} + \mathbf{O}\).
7. \(\tilde{\mathbf{h}} = \mathrm{RMSNorm}(\mathbf{h})\).
8. **SwiGLU:** \(\mathbf{a} = \sigma(\tilde{\mathbf{h}}\mathbf{W}_1) \odot (\tilde{\mathbf{h}}\mathbf{W}_3)\), \(\mathbf{f} = \mathbf{a}\mathbf{W}_2\) (shapes: \(\mathbf{W}_1,\mathbf{W}_3 \in \mathbb{R}^{d\times d_{\mathrm{ff}}}\), \(\mathbf{W}_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}\); \(\sigma=\mathrm{SiLU}\)).
9. **Residual:** \(\mathbf{y} = \mathbf{h} + \mathbf{f}\).

---

## 4. VARIABLES (LLaMA-3 8B Reference)

| Symbol | Typical value (8B class) | Role |
|--------|--------------------------|------|
| \(d\) | 4096 | Hidden size |
| \(H\) | 32 | Query heads |
| \(G\) | 8 | KV head **groups** |
| \(d_h\) | 128 | \(d/H\) |
| \(d_{\mathrm{kv}}\) | \(G\times d_h = 1024\) | Total K/V channel width before repeat |
| \(d_{\mathrm{ff}}\) | 14336 | SwiGLU intermediate width |
| \(L\) | variable (context length) | Sequence length |
| \(B\) | batch | Microbatch |
| \(N\) | 32 | Number of **decoder blocks** (depth) |
| \(|V|\) | 128256 (example) | Vocabulary for logits |
| \(\mathbf{W}_Q\) | \(d\times d\) | Query projection |
| \(\mathbf{W}_K,\mathbf{W}_V\) | \(d\times d_{\mathrm{kv}}\) | Compressed K,V |
| \(\mathbf{W}_O\) | \(d\times d\) | Attention output |
| \(\mathbf{W}_1,\mathbf{W}_3\) | \(d\times d_{\mathrm{ff}}\) | SwiGLU gate & up |
| \(\mathbf{W}_2\) | \(d_{\mathrm{ff}}\times d\) | SwiGLU down |

---

## 5. Layer Shapes and GQA

```
Configuration: d=4096, H=32, G=8, d_h=128, d_kv=1024, d_ff=14336

x, RMSNorm(x):        (B, L, 4096)
Q = x W_Q:            (B, L, 4096) → view (B, H, L, d_h)
K = x W_K:            (B, L, 1024) → view (B, G, L, d_h)  then repeat heads
V = x W_V:            (B, L, 1024) → view (B, G, L, d_h)

Attention weights:    (B, H, L, L)   after causal softmax
Context + W_O:        (B, L, 4096)

h = x + attn_out:     (B, L, 4096)

FFN SwiGLU:
  gate = SiLU(h' W1)   (B, L, d_ff)
  up   = h' W3         (B, L, d_ff)
  mid  = gate ⊙ up
  out  = mid W2        (B, L, 4096)

y = h + out:           (B, L, 4096)
```

**GQA:** \(H/G = 4\) query heads share one K,V slice — fewer K/V parameters and less KV-cache bandwidth than full MHA.

---

## 6. FLOPs per Decoder Block

FLOPs (multiply–add pairs): one matmul \(\mathbf{A}\in\mathbb{R}^{m\times k}\), \(\mathbf{B}\in\mathbb{R}^{k\times n}\) costs **\(2mkn\)** FLOPs.

**Per layer, causal self-attention (order \(\mathcal{O}(L^2)\) in \(L\)):**

| Term | FLOPs (leading terms) |
|------|----------------------|
| \(\mathbf{Q}\) | \(2BLd^2\) |
| \(\mathbf{K},\mathbf{V}\) | \(2 \times 2BLd\,d_{\mathrm{kv}} = 4BLd\,d_{\mathrm{kv}}\) |
| \(\mathbf{Q}\mathbf{K}^{\mathsf T}\) | \(2BL^2 d\) (since \(H d_h = d\)) |
| Softmax @ scores | Lower order \(\mathcal{O}(BHL^2)\) |
| \(\mathbf{A}\mathbf{V}\) | \(2BL^2 d\) |
| \(\mathbf{W}_O\) | \(2BLd^2\) |

**FFN SwiGLU:**

| Term | FLOPs |
|------|-------|
| \(\tilde{\mathbf{h}}\mathbf{W}_1\) | \(2BLd\,d_{\mathrm{ff}}\) |
| \(\tilde{\mathbf{h}}\mathbf{W}_3\) | \(2BLd\,d_{\mathrm{ff}}\) |
| \(\mathbf{a}\mathbf{W}_2\) | \(2BLd\,d_{\mathrm{ff}}\) |

**Approximate total per block:**

\[
\Phi_{\mathrm{block}} \approx
4BLd^2
+ 4BLd\,d_{\mathrm{kv}}
+ 4BL^2 d
+ 6BLd\,d_{\mathrm{ff}}
+ \text{(smaller terms)}.
\]

For **\(L \ll d\)** (short context), quadratic terms are small; for **\(L\) large** (e.g. 32k), **attention** dominates.

**Example:** \(B=1\), \(L=4096\), \(d=4096\), \(d_{\mathrm{kv}}=1024\), \(d_{\mathrm{ff}}=14336\).

- \(4Ld^2 \approx 4\cdot 4096 \cdot 4096^2 \approx 2.75\times 10^{11}\)
- \(4Ld\,d_{\mathrm{kv}} \approx 4\cdot 4096\cdot 4096\cdot 1024 \approx 6.87\times 10^{10}\)
- \(4L^2 d \approx 4\cdot 4096^2 \cdot 4096 \approx 2.75\times 10^{11}\) (attention on par with Q/O projections at this \(L\)!)
- \(6Ld\,d_{\mathrm{ff}} \approx 6\cdot 4096\cdot 4096\cdot 14336 \approx 1.46\times 10^{12}\) (**FFN dominates** at \(L=4096\) for typical LLaMA shapes)

So for **medium–long** \(L\), **SwiGLU** is often the largest slice of compute per layer.

---

## 7. Parameter Count per Layer

| Component | Formula (elements) | 8B-sized example |
|-----------|--------------------|------------------|
| \(\mathbf{W}_Q\) | \(d^2\) | \(4096^2 = 16{,}777{,}216\) |
| \(\mathbf{W}_K\) | \(d \cdot d_{\mathrm{kv}}\) | \(4096\cdot 1024 = 4{,}194{,}304\) |
| \(\mathbf{W}_V\) | \(d \cdot d_{\mathrm{kv}}\) | \(4{,}194{,}304\) |
| \(\mathbf{W}_O\) | \(d^2\) | \(16{,}777{,}216\) |
| \(\mathbf{W}_1,\mathbf{W}_3\) | \(2 d\,d_{\mathrm{ff}}\) | \(2\cdot 4096\cdot 14336 \approx 1.175\times 10^8\) |
| \(\mathbf{W}_2\) | \(d_{\mathrm{ff}}\,d\) | \(\approx 5.88\times 10^7\) |
| RMSNorm \(\boldsymbol{\gamma}\) (×2) | \(2d\) | negligible |

**Rough check:** Attention matrices \(\approx 4.2\times 10^7\) + FFN \(\approx 2.35\times 10^8\) per layer \(\Rightarrow\) **order \(2.8\times 10^8\)** parameters per block (plus small norms). Times **\(N=32\)** \(\Rightarrow\) **\(\sim 9\times 10^9\)** params in blocks alone; plus **embeddings** \(|V|\times d\) and **output head** (often **tied** to embedding).

---

## 8. Memory per Layer (Training Sketch)

**Inference (weights only, BF16):** one block’s weights \(\times 2\) bytes \(\approx 2.8\times 10^8 \times 2 \approx 560\) MB per layer in BF16 for the rough \(\sim 280\)M params above — *illustrative*; exact 8B counts follow official config tables.

**Training** adds (per-parameter, mixed-precision Adam-class):

| Copy | Precision | Role |
|------|-----------|------|
| Master weights | FP32 | Accurate accumulation |
| Optimiser states | FP32×2 (m,v) | Adam moments |
| Activations | BF16 (typical) | Forward + recomputed backward |

**Activations per layer** scale with **\(B\cdot L\cdot d\)** for main tensors; **attention maps** scale **\(B\cdot H\cdot L^2\)** if materialised (FlashAttention avoids full materialisation).

**Rule of thumb:** **FFN** layers use the most **parameters**; **attention** can use the most **activation memory** when \(L\) is huge unless IO-aware kernels are used.

---

## 9. Stacking 32 Blocks and the Full Pipeline

```
Token IDs:  t_1,...,t_L   (integers in [0, |V|))

     Embedding E:  (L, |V|) lookup  →  X in R^{L×d}
                         │
           ┌────────────┴────────────┐
           │  Block 0 (as in §3)      │
           └────────────┬────────────┘
                        │  repeat
                        ▼
           ┌─────────────────────────┐
           │  Blocks 1 … N−1         │
           └────────────┬────────────┘
                        ▼
                 RMSNorm_final (X_N)
                        │
                        ▼
          Logits = X_N W_head    often W_head = E^T  (weight tying)
                        │
                        ▼
               softmax(logits / T)   →  next-token distribution


For LLaMA-3 8B:  N = 32 decoder layers (+ embeddings + norms).
```

**Weight tying:** Sharing **embedding** and **LM head** reduces parameters by \(|V|\cdot d\) **if** you would otherwise store a separate head; capacity is slightly constrained but practice shows strong results.

---

## 10. NUMERICAL WALKTHROUGH

**Tiny dimensions:** \(B=1\), \(L=2\), \(d=4\), \(H=2\), \(d_h=2\), no GQA simplification — use full shapes for pedagogy.

Let \(\mathbf{x}\) be two tokens, \(\mathbf{x}_1=[1,0,0,1]\), \(\mathbf{x}_2=[0,1,1,0]\). Suppose after RMSNorm (with \(\gamma=\mathbf{1}\), \(\varepsilon\to 0\)) we obtain \(\hat{\mathbf{x}}_1 = \frac{1}{\sqrt{2}}[1,0,0,1]\), \(\hat{\mathbf{x}}_2 = \frac{1}{2}[0,1,1,0]\).

**Attention skip-path:** if \(\mathbf{W}_Q,\ldots\) initialise small, \(\mathbf{O}\approx \mathbf{0}\), then \(\mathbf{h} \approx \mathbf{x}\) early in training — **residual** preserves token-specific bias.

**FFN:** if \(\mathbf{W}_1,\mathbf{W}_3 \approx \epsilon\) small, SiLU gate \(\sigma(\cdot)\approx 0\), \(\mathbf{f}\approx \mathbf{0}\), \(\mathbf{y}\approx \mathbf{h}\); entire block \(\approx\) identity → stable start.

**FLOP micro-count:** One matmul \(\mathbb{R}^{2\times 4}\times\mathbb{R}^{4\times 4}\) costs \(2\cdot 2\cdot 4\cdot 4 = 64\) FLOPs. Scaling to LLaMA replaces \(4\to 4096\), \(2\to L\): dominant term often **FFN** \(6Ld\,d_{\mathrm{ff}}\).

---

## 11. Pseudocode

```python
def decoder_block(x, ln1, ln2, attn, ffn):
    """
    x: [B, L, d]  — same as LLaMA pre-norm block
    """
    h = x + attn(ln1(x))
    y = h + ffn(ln2(h))
    return y


def forward_llm(token_ids, embed, blocks, final_norm, lm_head):
    """
    token_ids: [B, L]; embed: nn.Embedding
    blocks: list of decoder_block modules
    """
    x = embed(token_ids) * sqrt(d_model)   # optional GPT/Llama scaling
    for block in blocks:
        x = block(x)
    x = final_norm(x)
    logits = lm_head(x)                    # [B, L, |V|]
    return logits
```

**RoPE, causal mask, GQA head repeat** are inside `attn` in real code.

---

## 12. COMMON MISTAKES

| ❌ Wrong | ✓ Correct |
|---------|-----------|
| “Attention always dominates FLOPs.” | For large \(d_{\mathrm{ff}}\) and moderate \(L\), **FFN** often wins; for huge \(L\), **attention** wins. Profile with your \(L,d,d_{\mathrm{ff}}\). |
| “GQA changes softmax shape.” | Softmax is still \((B,H,L,L)\); **K,V are smaller** and **broadcast** to query heads. |
| “Post-norm LLaMA.” | LLaMA is **pre-norm**: residual is **outside** norm; \(\mathrm{FFN}(\mathrm{RMSNorm}(h))\) not \(\mathrm{RMSNorm}(h+\mathrm{FFN}(\cdot))\). |
| “Parameter count = FLOPs.” | Params \(\neq\) ops; FFN has **two** large up-projs (gate+up) but **one** attention \(W_Q\) of size \(d^2\). |

---

## 13. EXERCISES

1. **Derive** \(\Phi_{\mathrm{block}}\) for your target \(L,d,H,G,d_{\mathrm{ff}},B\) and state which term wins.

2. **GQA:** With \(H=32\), \(G=8\), how many distinct **K,V slices** per token per layer? How many compared with full MHA?

3. **Memory:** Estimate activation bytes for one layer’s **attention matrix** \((B,H,L,L)\) in FP16 vs using **FlashAttention** (no full materialisation — qualitative).

4. **Tying:** If \(|V|=128256\), \(d=4096\), how many parameters in **embedding**? If head is untied, how many **additional** params?

5. **Depth:** Show that output after \(N\) identical identity blocks equals input; explain why **learned** \(\mathcal{F}_\ell\) break symmetry across layers.

---

**References:** Touvron et al., LLaMA; Dubey et al., The LLaMA 3 Herd of Models (2024); Vaswani et al., Attention Is All You Need; Shazeer, GLU Variants (SwiGLU).
