# QLoRA — Quantized LoRA for Memory-Efficient Fine-Tuning

This note formalizes **QLoRA**: base weights stored in **4-bit Normal Float (NF4)**, low-rank adapters trained in **BF16**, **double quantization** of scales, **paged optimizers**, memory accounting, and why **NF4** is effective for Gaussian-like matrices. Includes training recipe and indicative memory comparisons.

---

## Table of Contents

- [1. Variables](#1-variables)
- [2. Intuition — Frozen 4-bit Base, High-precision Adapters](#2-intuition--frozen-4-bit-base-high-precision-adapters)
- [3. LoRA Recap (Low-Rank Update)](#3-lora-recap-low-rank-update)
- [4. NF4 and Information-Theoretic Optimality for Gaussian Weights](#4-nf4-and-information-theoretic-optimality-for-gaussian-weights)
- [5. Double Quantization](#5-double-quantization)
- [6. Paged Optimizers and Memory Spikes](#6-paged-optimizers-and-memory-spikes)
- [7. Memory Footprint Formula](#7-memory-footprint-formula)
- [8. Training Recipe (Practical Pseudocode)](#8-training-recipe-practical-pseudocode)
- [9. Memory Comparison — Full FT vs LoRA vs QLoRA](#9-memory-comparison--full-ft-vs-lora-vs-qlora)
- [10. Numerical Examples](#10-numerical-examples)
- [Appendix: Lloyd-Max link (qualitative)](#appendix-lloyd-max-link-qualitative)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)

---

## 1. Variables

| Symbol | Meaning |
|--------|---------|
| \(W \in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\) | Original linear weight matrix |
| \(Q(W)\) | Quantized storage of base weights (e.g., NF4 blocks + scales) |
| \(A \in \mathbb{R}^{r\times d_{\mathrm{in}}},\, B \in \mathbb{R}^{d_{\mathrm{out}}\times r}\) | LoRA factors, \(r \ll \min(d_{\mathrm{in}},d_{\mathrm{out}})\) |
| \(\alpha\) | LoRA scaling hyperparameter (\(\Delta W = (\alpha/r)\,B A\) in common conventions) |
| \(b\) | Quantization bitwidth for base weights (here \(b=4\)) |
| \(s,\, z\) | Per-block **scale** and **zero-point** (or bias) used to map codes to floats |
| \(M\) | Total GPU memory footprint (order-of-magnitude accounting) |
| \(\Theta_{\mathrm{LoRA}}\) | Trainable adapter parameters only |
| **NF4** | Normal Float 4-bit: 16-bin quantisation tailored near \(\mathcal{N}(0,1)\) |

---

## 2. Intuition — Frozen 4-bit Base, High-precision Adapters

```
FULL FINE-TUNE (expensive):
  W  (FP16/BF16)  ──optimize──►  W'     (every element moves)

LoRA (lighter):
  W  (FP/BF16)  frozen |||||||||||
                ΔW = B·A  (tiny rank r)  ──optimize──►  A,B only

QLoRA (lighter still):
  Q(W)  NF4 in VRAM  ─frozen, dequant on-the-fly in matmul────►  hidden states
            │
            └──►  + (α/r) B A   (BF16 adapters trained)
```

**Intuition:** Most “task signal” lives in a **low-dimensional subspace** of updates (LoRA hypothesis). QLoRA compresses the **massive** base in NF4 but keeps **adapter dynamics** in high enough precision that optimisation remains well-conditioned.

---

## 3. LoRA Recap (Low-Rank Update)

Forward pass of a single linear (ignoring biases):

\[
y = W x + \frac{\alpha}{r}\, B A x,\qquad \mathrm{rank}(BA)\le r.
\]

**Gradients:**

- If \(W\) is frozen, \(\partial \mathcal{L}/\partial W = 0\) (in true freeze).
- Gradients flow to \(A,B\) through the adapter path; **activations** \(x\) still **pass through dequantized** \(W\) in QLoRA.

**Computational graph:**

\[
x \xrightarrow{\text{dequant } \hat{W}} \hat{W}x \;+\; \frac{\alpha}{r} BAx.
\]

Here \(\hat{W}\) is **not stored** in FP16; it is reconstructed per kernel from codes + scales.

---

## 4. NF4 and Information-Theoretic Optimality for Gaussian Weights

Empirically, many Transformer weights exhibit **approximately unimodal, bell-shaped** distributions after layer normalization and scaling. A coarse model is \(w \sim \mathcal{N}(0,\sigma^2)\).

### 4.1 Equal-probability bins on the Gaussian

Let \(\Phi\) be the CDF of \(\mathcal{N}(0,1)\). Partition \(\mathbb{R}\) into 16 intervals \(\{I_1,\ldots,I_{16}\}\) with equal mass:

\[
\mathbb{P}(w \in I_k) = \frac{1}{16}.
\]

**Lloyd-type centroid quantisation** chooses representative levels \(q_k\) to minimise MSE:

\[
\min_{\{q_k\}} \mathbb{E}\bigl[(w - q(w))^2\bigr],\quad q(w)\in\{q_1,\ldots,q_{16}\}.
\]

For equal-probability bins under a smooth density, **conditional expectations** \(q_k = \mathbb{E}[w \mid w\in I_k]\) **minimize bin-wise squared error** structure; NF4 uses a **fixed** 16-level codebook designed around \(\mathcal{N}(0,1)\) **then applies a per-tensor scale** to fit empirical spread.

### 4.2 Why “Gaussian-optimality” is plausible

- If weights are **exactly** Gaussian, fixed **equal-density** bins + centroid levels minimise **mean squared quantisation error** under certain regularity assumptions (Lloyd-Max intuition).
- Real networks deviate: outliers, asymmetric layers. In practice, **blockwise scales** restore dynamic range.

**ASCII sketch (symmetric binning idea):**

```
   p(w)

     ▲                    each bin has equal area (=equal mass)
     │      ╱‾‾╲
     │    ╱      ╲
     │  ╱          ╲___
     └──────────────────────► w
        |   |     |   |      (16 bins → 16 NF4 codes)
```

---

## 5. Double Quantization

**First quantisation:** map each **block** (or channel group) of weights to 4-bit indices + **FP16/BF16 scale** \(s\).

**Problem:** millions of small scales \(s\) themselves consume FP16 memory.

**Double quantisation:** quantise the **scales** **again** (e.g., 8-bit) so metadata overhead shrinks:

\[
s \xrightarrow{Q^{(2)}} \mathrm{code}(s)\quad(\text{with its own scale/offset}).
\]

**Informal memory effect:** Base weight codes dominate; DQ trims a **noticeable** overhead fraction especially at large widths.

---

## 6. Paged Optimizers and Memory Spikes

AdamW keeps **first and second moments** per trainable parameter (FP32 in many implementations):

\[
m_t,\; v_t \in \mathbb{R}^{|\Theta|}.
\]

During long sequences or large microbatch token counts, transient activations may peak VRAM usage. **Paged optimizers** offload optimizer states to **CPU RAM** when GPU memory is **pressure-spiking**, then page chunks back—trading slower steps for survival within budget.

**When it matters:** long context + MoE + adapter training on **consumer GPUs**.

---

## 7. Memory Footprint Formula

A **conceptual** decomposition (train QLoRA, adapters only in Adam FP32):

\[
M \approx \underbrace{\mathrm{NF4\_weights}}_{\text{base tensors in 4-bit}}
\;+\;\underbrace{\mathrm{BF16\_adapters}}_{\text{trainable }A,B}
\;+\;\underbrace{\mathrm{FP32\_optimizer}(\Theta_{\mathrm{LoRA}})}_{\text{Adam }m,v \text{ for adapters}}.
\]

**Important:** Base weights **gradients** are not materialized in full precision in standard QLoRA training (base frozen), which is why this is far smaller than full fine-tuning.

**Order symbols (per parameter counts):**

- Full finetune train states: often **FP16/BF16 weights + FP32 momenta** for **all** parameters.
- QLoRA: **4-bit base weights + BF16 adapters + FP32 moments for adapters only**.

---

## 8. Training Recipe (Practical Pseudocode)

```
load base model into 4-bit (NF4 + double quant of scales)
inject LoRA modules on chosen projections (q,v,k,o / MLP)
optimizer = AdamW(θ_LoRA, lr=η, wd=λ)

for minibatch in D:
    x, y_mask = batch
    # forward uses dequant matmul kernels for base + adapter addition
    logits = forward_qlora(model, x)
    loss = masked_ce(logits, y, y_mask)
    loss.backward()        # gradients land on adapter params only
    optimizer.step()

    # optional: gradient checkpointing on activations
    # optional: paged optimizer handles CPU paging transparently
```

**Hyperparameter anchors (order-of-magnitude, not universal):**

- Rank \(r\in\{8,16,64,128\}\) depending on target complexity.
- \(\alpha\) tied to \(r\) (e.g., \(\alpha=16\) or \(\alpha=2r\); follow library defaults consistently).
- LR often **similar to LoRA on FP16 base**, but validate—quant noise path differs.

---

## 9. Memory Comparison — Full FT vs LoRA vs QLoRA

Let \(N\) be total base parameters. Ignoring embeddings, KV-cache, activations (context-dependent):

| Mode | Dominant weight storage | Dominant trainable states |
|------|-------------------------|---------------------------|
| Full FT | BF16/FP16 **all** \(N\) | Adam FP32 **\(\sim 2N\)** often |
| LoRA (BF16/FP16 base) | BF16 base | BF16 adapters + Adam on adapters |
| QLoRA | **NF4 base (~0.5 bytes/param class memory)** | BF16 adapters + Adam on adapters |

**Indicative (vary widely by implementation):**

- **7B** params: full FT commonly needs **40–60+ GB** class with optimizer states; QLoRA setups often target **\(<24\) GB** consumer training depending on sequence length and framework overhead.
- **13B**: full FT typically **\(\sim 2\times\)** needs vs 7B; QLoRA gap widens vs full FT.
- **70B**: full FT frequently **multi-GPU**; QLoRA enables **single high-end consumer GPU** training of adapters in favorable configurations (still tight on context length).

**Why tables differ across blogs:** KV cache, activation checkpointing, flash-attention, gradient accumulation batching, and whether **output embeddings** are trained all move totals.

---

## 10. Numerical Examples

### Example 1 — Bit accounting (idealized)

Naïve BF16 base storage \(\approx 2\) bytes/param. NF4 \(\approx 0.5\) bytes/param **for codes only**—**4×** compression of raw weights **before** scales; with block scales + DQ, realised savings are **moderately less** than **4×** total DRAM footprint.

### Example 2 — Adapter parameter count

If a layer has \(d_{\mathrm{out}}=4096, d_{\mathrm{in}}=4096\), full \(W\) has \(16{,}777{,}216\) entries. With \(r=16\),

\[
|A|+|B| = r d_{\mathrm{in}} + r d_{\mathrm{out}} = 16\times 8192 = 131{,}072,
\]

**\(\sim 128\times\)** fewer trained elements in that layer’s adapter versus dense \(W\) (before counting multiple projections per layer).

### Example 3 — Optimizer memory on adapters only

If total trainable adapters are \(10^7\) parameters, Adam FP32 states \(\approx 2\times 4\times 10^7\) bytes \(\approx 80\) MB **just** for moments (order-of-magnitude). Compare to storing **full** FP32 moments for 7B (**tens of GB**).

### Example 4 — Blockwise affine map (dequantisation)

Partition \(W\) into blocks \(B_j\) of size \(B\) (e.g., 64 or 128 elements per block). For block \(j\), NF4 **indices** \(c_i\in\{0,\ldots,15\}\) and **scale** \(s_j\in\mathbb{R}_{>0}\) reconstruct

\[
\hat{w}_{j,i} = s_j \cdot \mathrm{NF4\_centroid}(c_{j,i}).
\]

Thus each block is an **affine pushforward** of a discrete code—gradients through adapters do **not** backprop into discrete \(c\) (frozen \(|\) straight-through estimators absent), but **forward activations** use \(\hat{w}\) as data.

### Example 5 — FLOPs vs memory binding

Suppose a layer matmul costs \(\mathcal{O}(d_{\mathrm{out}} d_{\mathrm{in}})\) MACs. LoRA adds \(\mathcal{O}(r(d_{\mathrm{in}}+d_{\mathrm{out}}))\) extra MACs versus \(\mathcal{O}(r d_{\mathrm{in}} + r d_{\mathrm{out}})\) for two low-rank matmuls—cheap when \(r\ll d\). QLoRA’s dominant question is whether **dequant** bandwidth stalls the GPU; vendor kernels fuse **unpack + matmul**.

---

## Appendix: Lloyd-Max link (qualitative)

For a 1D density \(f(w)\), optimal quantisation levels \(\{q_k\}\) for \(K\) bins with boundaries \(\{b_k\}\) satisfy **centroid condition**:

\[
q_k = \frac{\int_{b_{k-1}}^{b_k} w\, f(w)\,\mathrm{d}w}{\int_{b_{k-1}}^{b_k} f(w)\,\mathrm{d}w}.
\]

For **uniform bin masses** on a Gaussian, boundaries are **equispaced quantiles**, centroids are NF4-like. Real NF4 is a **frozen** codebook + **learned per-block** \(s_j\) to handle **non-Gaussian** empirical tails.

---

## 11. Common Mistakes

- ❌ **Assume NF4 alone reduces optimizer memory for full weights.**

  ✓ QLoRA savings assume **adapters-only** training (or selective finetuning). Full-parameter Adam still explodes memory.

- ❌ **Confuse “4-bit storage” with “4-bit compute precision everywhere”.**

  ✓ Many kernels **promote to higher precision** for accumulator math; training stability depends on **BF16 adapters**.

- ❌ **Set rank \(r\) huge “to be safe.”**

  ✓ Larger \(r\) increases adapter compute/memory quadratically in \(r\) across projections—validate on dev.

- ❌ **Ignore double quant overhead accounting** in custom stacks.

  ✓ Measure with profiling; metadata can matter at huge widths.

- ❌ **Expect identical loss curves** as BF16 LoRA.

  ✓ Quant noise + kernel differences → slightly different calibration—may need LR sweep.

---

## 12. Exercises

1. **Rank budget:** For a single linear, express parameter count of LoRA versus full \(W\). Solve for \(r^\*\) where LoRA matches dense count.

2. **MSE intuition:** For a 1D Gaussian, write the Lloyd-Max fixed-point iteration for two levels. Generalize qualitatively to 16 levels.

3. **Scaling law for \(\alpha/r\):** Show how multiplying \(B\) by \(c\) and dividing \(A\) by \(c\) leaves \(BA\) unchanged—why is \(\alpha\) still needed?

4. **Memory formula:** Estimate bytes for NF4 weights \(\approx 0.5N\) + BF16 adapters \(2|{\Theta_{\mathrm{LoRA}}}|\) + FP32 optimizer \(8|{\Theta_{\mathrm{LoRA}}}|\) under 2 moment vectors.

5. **Paged optimizers:** Under what step latency inequality is paging worth it?

---

### Closing remark

QLoRA is a **systems–learning hybrid**: the mathematics is still **conditional likelihood** (or whatever task loss), but **implementation kernels** determine whether the method is stable. Always benchmark **tokens/sec** and **loss** jointly—not bytes alone.
