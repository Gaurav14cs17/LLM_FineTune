# Mixed Precision Training for LLMs

*BF16 vs FP16, master weights, loss scaling, memory copies, and FP8 precisions*

---

## Table of Contents

1. [Overview](#1-overview)
2. [INTUITION: Bits Budget](#2-intuition-bits-budget)
3. [VARIABLES](#3-variables)
4. [IEEE-754 Style Layouts: FP32, BF16, FP16](#4-ieee-754-style-layouts-fp32-bf16-fp16)
   - [INTUITION: Bit Packing](#41-intuition-bit-packing)
5. [Why BF16’s Exponent Range Matters](#5-why-bf16s-exponent-range-matters)
6. [Training Stack: FP32 Master + BF16 Compute](#6-training-stack-fp32-master--bf16-compute)
7. [Loss Scaling for FP16 (Not BF16)](#7-loss-scaling-for-fp16-not-bf16)
8. [The Three-Copy Memory Picture](#8-the-three-copy-memory-picture)
9. [FP8: E4M3 and E5M2](#9-fp8-e4m3-and-e5m2)
10. [Numerical Stability Analysis](#10-numerical-stability-analysis)
11. [NUMERICAL EXAMPLES](#11-numerical-examples)
12. [Pseudocode](#12-pseudocode)
13. [COMMON MISTAKES](#13-common-mistakes)
14. [EXERCISES](#14-exercises)

---

## 1. Overview

**Mixed precision** trains large models using **low-precision** (16-bit or 8-bit) **tensor arithmetic** for speed and memory, while keeping **high-precision** copies where **round-off** would break optimisation. BF16 is the **default** for many LLM pretrain runs on NVIDIA hardware; FP16 often needs **dynamic loss scaling**; **FP8** is emerging for next-gen kernels.

This note tabulates **bit layouts**, compares **dynamic range** and **mantissa precision**, explains **master weights** + **Adam states**, sketches the **three memory copies**, introduces **FP8** variants, and analyses **underflow/overflow** failure modes.

---

## 2. INTUITION: Bits Budget

```
┌────────────────────────────────────────────────────────────────────┐
│  Fixed 16 bits — you choose what to preserve:                      │
│                                                                    │
│    WIDE EXPONENT  →  huge dynamic range, FEW mantissa bits          │
│    WIDE MANTISSA  →  fine precision, NARROW exponent (risk clip)   │
│                                                                    │
│     BF16:  same exponent width as FP32  (8 bits)                   │
│            ↳ range like FP32  →  activations rarely clip           │
│            ↳ only 7 mantissa bits  →  RELATIVE precision coarse    │
│                                                                    │
│     FP16: 5-bit exponent, 10-bit mantissa                          │
│            ↳ MORE precise mantissa than BF16                       │
│            ↳ SMALL max (~65504)  →  overflow in matmuls common      │
│                                                                    │
│  LLM training prefers range (BF16) over mantissa finesse FP16    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(S\) | Sign bit |
| \(E\) | Exponent field width (bits) |
| \(M\) | Mantissa (fraction) width (bits) |
| bias | Exponent bias (format-dependent) |
| \(S_{\mathrm{LS}}\) | Loss scale for FP16 (dynamic) |
| \(\mathbf{W}_{32}\) | FP32 **master** weights |
| \(\mathbf{G}_{16}\) | BF16/FP16 **gradients** |
| \(\mathbf{m},\mathbf{v}\) | Adam first/second moments (typically FP32) |

---

## 4. IEEE-754 Style Layouts: FP32, BF16, FP16

### 4.1 INTUITION: Bit Packing

```
           FP32  (32 bits)
           ┌─┬────────┬───────────────────────────┐
           │S│  Exp   │        Mantissa         │
           └─┴────────┴───────────────────────────┘
            1   8 bits          23 bits

           BF16  (16 bits)  — "truncate" FP32 mantissa
           ┌─┬────────┬──────────────┐
           │S│  Exp   │   Mantissa   │   exponent SAME width as FP32
           └─┴────────┴──────────────┘
            1   8 bits      7 bits

           FP16  (16 bits)  — rebalanced
           ┌─┬─────┬──────────────┐
           │S│ Exp │   Mantissa   │   5 exp / 10 mant
           └─┴─────┴──────────────┘
            1  5 bits   10 bits
```

**Normal numbers** (conceptual):

\[
(-1)^S \times 2^{e-\mathrm{bias}} \times (1 + m/2^M),
\]

with **stored** exponent bits encoding \(e\).

| Format | S | E (exp bits) | M (mant bits) | Total | Approx max normal | Notes |
|--------|---|--------------|---------------|-------|-------------------|-------|
| FP32 | 1 | 8 | 23 | 32 | \(\sim 3.4\times 10^{38}\) | reference |
| BF16 | 1 | 8 | 7 | 16 | \(\sim 3.4\times 10^{38}\) | truncates mantissa of FP32 |
| FP16 | 1 | 5 | 10 | 16 | **65504** | much smaller max |

**BF16:** Same **exponent** interpretation as FP32 (bias 127); mantissa is **truncated** from FP32, not renormalised with a new layout.

**FP16:** 5-bit exponent (bias 15), 10-bit mantissa — **different tradeoff**: higher relative precision near 1, **tiny** absolute max compared to BF16.

---

## 5. Why BF16’s Exponent Range Matters

Large matmuls accumulate **inner products** with magnitude \(\mathcal{O}(\sqrt{d}\cdot \sigma_x \sigma_w)\) in rough scaling. With \(d=4096\) and occasional large activations, **FP16** saturates (\(\pm 65504\)) → **Inf** in forward or backward → **NaN** loss.

**BF16** rarely overflows to Inf for typical LLM ranges because its **ceiling** matches FP32-class exponents.

Tiny gradients: **subnormals** and **underflow** differ by format; BF16 **normal** minimum \(\approx 1.75\times 10^{-38}\) (same order as FP32 normals — **not** FP16-tiny).

---

## 6. Training Stack: FP32 Master + BF16 Compute

**Forward:**

1. Cast \(\mathbf{W}_{32} \rightarrow \mathbf{W}_{\mathrm{bf16}}\) (on-the-fly or cached).
2. Compute \(\mathbf{a}_{\ell} = \mathrm{Layer}_\ell(\mathbf{a}_{\ell-1}, \mathbf{W}_{\mathrm{bf16}})\) in BF16 Tensor Cores.

**Backward:**

1. Gradients \(\mathbf{G}_{\mathrm{bf16}}\) in BF16.
2. **Upcast** to FP32 for **accumulation** into \(\mathbf{G}_{32}\) before **optimiser**.

**Optimiser (AdamW):**

\[
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_{32},\quad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_{32}^2,
\]
\[
\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^t),\quad \hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t),
\]
\[
\mathbf{W}_{32} \leftarrow \mathbf{W}_{32} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \mathbf{W}_{32}
\]
(all in **high precision** for stability).

---

## 7. Loss Scaling for FP16 (Not BF16)

FP16 gradients can **underflow** (become **zero**) because the **smallest normals** are \(\sim 6\times 10^{-5}\) and subnormals \(\sim 10^{-8}\) order.

**Dynamic loss scaling** (Micikevicius et al., 2018):

1. Multiply loss by **`S`** (e.g. \(2048\)): \(\tilde{\mathcal{L}} = S \mathcal{L}\).
2. Backward yields \(\tilde{\mathbf{g}} = S \mathbf{g}\) in regions where autodiff is linear in \(\mathcal{L}\).
3. Before Optimizer: \(\mathbf{g} = \tilde{\mathbf{g}} / S\).

Adjust **`S`**: if **Inf/NaN**, skip step, **halve** `S`; if many good steps, **double** `S` slowly.

**BF16:** Wide exponent → **loss scaling usually unnecessary** — simplifies training stack.

---

## 8. The Three-Copy Memory Picture

For a parameter tensor, **training** commonly holds:

| Copy | Precision | Purpose |
|------|-----------|---------|
| **Master weights** | FP32 | Small updates accumulate faithfully |
| **Active / forward weights** | BF16 | Matmuls, Tensor Cores |
| **Gradients** | BF16 | Backprop; then upcast |

Plus **Adam**: \(\mathbf{m},\mathbf{v}\) in **FP32** → **two more** FP32 tensors per parameter in standard impl.

**Rough bytes / param (Adam + BF16 grads + FP32 master):**

- FP32 master: **4**
- FP32 m: **4**
- FP32 v: **4**
- BF16 grad: **2**

≈ **14 B/param** **order-of-magnitude** before activations (excludes optimiser fusion / 8-bit optim).

**Example (8B weights, pure accounting):** \(8\times 10^9 \times 14\) B \(\approx 112\) GB for master+moments+BF16 grad **before** embeddings-only optimiser tricks — real stacks add **activation checkpointing**, **ZeRO**, or **8-bit Adam** to cut this.

---

## 9. FP8: E4M3 and E5M2

**FP8** (\(\texttt{float8}\)) trades range/precision in **8 bits**:

| Format | Sign | Exp | Mantissa | Typical use |
|--------|------|-----|----------|-------------|
| **E4M3** | 1 | 4 | 3 | Weights & activations (more precision) |
| **E5M2** | 1 | 5 | 2 | Gradients (wider exponent, coarser mantissa) |

**E4M3** max \(\sim 448\): **narrow** vs BF16 — requires **careful scaling** (block scaling, tensor scaling) in frameworks.

Hardware (H100, etc.) accelerates FP8 GEMMs; **recipes** evolve with libraries (Transformer Engine, etc.).

---

## 10. Numerical Stability Analysis

### 10.1 Rounding vs Deep Training

BF16 has \(\sim 7\) fraction bits \(\Rightarrow\) **unit roundoff** \(u \approx 2^{-7}\) relative to significand, much coarser than FP32. **Error accumulation** over \(T\) steps is **not** simply \(T\cdot u\) (gradients fluctuate, signs cancel), but **systematic bias** in Adam can occur if small updates always round the same way — **FP32 master** amortises this.

### 10.2 Reduction Kernels

**1. Weight updates:** \(\Delta \mathbf{W} \sim \eta \mathbf{g}/(\sqrt{\hat{\mathbf{v}}}+\epsilon)\). If \(\mathbf{g}\) is BF16, **very small** updates can round to **zero** before hitting FP32 master — mitigated by **FP32 grads** in some stacks or **Stochastic rounding** (advanced).

**2. Reductions:** LayerNorm / softmax accumulate in **FP32** in high-quality kernels (even if IO is BF16).

**3. Adam \(\epsilon\):** Usually \(\sim 10^{-8}\) avoids division blow-ups; lives in FP32.

### 10.3 Tensor Cores and Throughput

NVIDIA **Tensor Cores** execute **WMMA**-style ops in **BF16/FP16** at **many×** the FP32 CUDA-core rate on datacentre GPUs. That is the **hardware** reason mixed precision dominates LLM training: **matmul-bound** steps become cheaper; memory bandwidth can become the **next** bottleneck (hence FlashAttention, sequence packing).

**4. BF16 + large LR:** Can still diverge — mixed precision **does not** replace **schedule discipline**.

---

## 11. NUMERICAL EXAMPLES

**Example A — BF16 mantissa spacing (order \(2^{-7}\) relative to significand scale):** Near value \(1.0\), spacing \(\Delta x \sim 2^{-7}\). An update \(\Delta w = 10^{-5}\) **larger** than spacing → representable after accumulation in **FP32** master.

**Example B — FP16 max:** Value \(10^5\) exceeds FP16 max → **Inf** if tensor not scaled.

**Example C — Gradient \(g = 2\times 10^{-8}\):** Compare roughly to FP16 normal min \(\sim 6\times 10^{-5}\): **underflows to 0** as normal; **BF16** (FP32-class range) represents small normals — **no forced zero** for this magnitude as in FP16 pathological case.

**Example D — Loss scale:** True \(g=10^{-9}\). With \(S=2048\), \(\tilde{g}\sim 2\times 10^{-6}\), still small but **less likely** to flush in FP16.

---

## 12. Pseudocode

```python
# Mixed precision training (schematic)

def forward_backward(model_fp32_master, data, loss_scaler=None):
    # cast weights to bf16 for matmuls inside modules
    loss = model_bf16_forward(model_fp32_master, data)  # BF16 compute

    if loss_scaler is not None:
        loss = loss * loss_scaler.get_scale()

    loss.backward()   # BF16 grads in weights

    if loss_scaler is not None:
        loss_scaler.unscale_(optimizer)  # divide grads by S inside optim step
        # detect inf, skip step, adjust S

    # optimizer step on FP32 master + FP32 moments
    optimizer.step()
    optimizer.zero_grad()
```

For **BF16-only**, `loss_scaler` can be **disabled** in many LLM recipes.

---

## 13. COMMON MISTAKES

| ❌ Wrong | ✓ Correct |
|---------|-----------|
| “BF16 = FP16.” | Different **exponent** width → different **overflow** behaviour. |
| “No loss scaling → no training issues.” | Still need **FP32 master**, good **LR**, sometimes **grad clip**. |
| “Three copies always fit in HBM.” | **Activations** often dominate; **checkpointing** changes footprint. |
| “FP8 drop-in identical to BF16.” | FP8 needs **scaling** strategies; **max** for E4M3 ~448. |

---

## 14. EXERCISES

1. Estimate total **optimiser + weight** memory for **8B** params: FP32 master, FP32 m,v, BF16 grads (ignore activations).

2. Explain **why** dynamic loss scaling multiplies **loss** rather than **post-hoc** doubling gradients manually in each layer.

3. Compare **smallest positive normal** for FP16 vs BF16 using exponent bias formulas (look up IEEE constants).

4. **E4M3**: If tensor values exceed **448**, what happens without scaling? Propose mitigation at module boundary.

5. **Mixed precision + AdamW:** Which tensors **must** stay FP32 for billion-step runs and why?

6. **Forward-only inference:** Explain why **FP16/BF16 weights only** (no master) are common for deployment, while training keeps FP32 master.

7. Derive order-of-magnitude for **relative** quantization error \(\Delta x / x\) for BF16 vs FP16 **near** \(x=1\) using mantissa bit counts.

---

**References:** Micikevicius et al., Mixed Precision Training (2018); NVIDIA A100/H100 docs; `torch.cuda.amp`, Transformer Engine; Kalamkar et al., Study of BFloat16 (2019); FP8 formats (NVIDIA, ARM, Graphcore whitepapers).
