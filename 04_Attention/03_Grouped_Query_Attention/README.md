# Understanding Grouped Query Attention (GQA)

*KV-head sharing between multi-head attention (MHA) and multi-query attention (MQA): memory, algebra, and LLaMA‑3 scale*

---

**Grouped Query Attention (GQA)** partitions query heads into \(G\) groups that **share** a single key–value projection pair per group. Between the extremes **MHA** (\(h_{\mathrm{kv}} = h_q\)) and **MQA** (\(h_{\mathrm{kv}} = 1\)), GQA (\(1 < h_{\mathrm{kv}} < h_q\)) recovers most representational flexibility while shrinking the **KV cache** by the factor \(h_q / h_{\mathrm{kv}}\)—for LLaMA‑3 8B, \(h_q=32\), \(h_{\mathrm{kv}}=8\) gives a **4×** KV reduction versus MHA at inference.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 MHA, MQA, and GQA at a Glance](#11-mha-mqa-and-gqa-at-a-glance)
   - [1.2 Why the KV Cache Dominates Inference](#12-why-the-kv-cache-dominates-inference)
2. [Mathematical Formulation](#2-mathematical-formulation)
   - [2.1 Attention with Shared KV Heads](#21-attention-with-shared-kv-heads)
   - [2.2 Parameter and Activation Tensor Shapes](#22-parameter-and-activation-tensor-shapes)
3. [Memory and KV Cache Formulas](#3-memory-and-kv-cache-formulas)
   - [3.1 Per-Token KV Bytes](#31-per-token-kv-bytes)
   - [3.2 Reduction Factor \(h_q / h_{\mathrm{kv}}\)](#32-reduction-factor-h_q--h_mathrmkv)
   - [3.3 Comparison Table (MHA vs MQA vs GQA)](#33-comparison-table-mha-vs-mqa-vs-gqa)
   - [3.4 Decode-Step Attention FLOPs (Informal)](#34-decode-step-attention-flops-informal)
4. [Why Grouping Works (Derivation Sketch)](#4-why-grouping-works-derivation-sketch)
   - [4.1 Algebraic View of Shared K, Diverse α](#41-algebraic-view-of-shared-k-diverse-alpha)
   - [4.2 Why Not Always MQA?](#42-why-not-always-mqa)
5. [LLaMA‑3 Setting and Numerical Example](#5-llama3-setting-and-numerical-example)
6. [Checkpoint Conversion (Mean Pooling)](#6-checkpoint-conversion-mean-pooling)
7. [Pseudocode](#7-pseudocode)
8. [Common Mistakes](#8-common-mistakes)
9. [Exercises](#9-exercises)

---
## 1. Overview

### ┌─────────────────────────────────────────────────────────────┐
### │ INTUITION: attention as many “questions,” fewer “indexes”    │
### └─────────────────────────────────────────────────────────────┘

```
        MHA                         GQA (e.g. h_q=32, h_kv=8)
   ┌───┬───┬───┬───┐                 ┌───────────────┐  K₁ ──┐
Q: │Q₁ │Q₂ │… │Q₃₂│                 │ Q₁…Q₄  share  │  K₂ ──┼──► softmax rows
   └───┴───┴───┴───┘                 │ Q₅…Q₈  share  │  …    │
   │ │ │ │   (each own K,V)          └───────────────┘  K₈ ──┘
   K₁V₁ K₂V₂ … K₃₂V₃₂                    8 K/V pairs serve 32 Q heads

        MQA (extreme)
   ┌───┬───┬───┬───┐
Q: │Q₁ │Q₂ │… │Q₃₂│   ──►  ONE K, ONE V for ALL queries  (max memory save, max tie)
   └───┴───┴───┴───┘
```

**Idea:** Queries can diversify **routing** (what to look for), while keys/values provide a **compressed index** of context. Sharing \(K,V\) across a **group** of queries asks many related questions of the same contextual “evidence”—an empirically strong trade-off when \(h_{\mathrm{kv}}\) is not too small.

### 1.1 MHA, MQA, and GQA at a Glance

| Variant | \(h_q\) (query heads) | \(h_{\mathrm{kv}}\) (K/V head groups) | KV vs MHA |
|--------|------------------------|----------------------------------------|-----------|
| MHA | \(h\) | \(h\) | \(1\times\) |
| GQA | \(h\) | \(G\) | \((h/G)\times\) smaller KV |
| MQA | \(h\) | \(1\) | \(h\times\) smaller KV |

### 1.2 Why the KV Cache Dominates Inference

During **autoregressive** decoding, each new token appends one query row but must attend over **all past** keys and values. Storing \(K_t, V_t\) for all positions \(t \le T\) across \(L\) layers drives **memory** and caps **batch size**. Shrinking the **width** of cached \(K,V\) in the head dimension (fewer distinct \(K,V\) projections) directly reduces bytes per token.

---

## 2. Mathematical Formulation

### VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(h_q\) | Number of query heads (often denoted \(h\)) |
| \(h_{\mathrm{kv}}\) | Number of key–value head groups (LLaMA‑3 8B: **8**) |
| \(g = h_q / h_{\mathrm{kv}}\) | Query heads **per** KV group (LLaMA‑3 8B: \(32/8 = 4\)) |
| \(d_{\text{model}}\) | Hidden size \(d\) (e.g. **4096**) |
| \(d_k, d_v\) | Per-head dimensions; often \(d_k = d_v = d_{\text{model}}/h_q\) |
| \(B\) | Microbatch size (concurrent sequences) |
| \(T\) | Sequence length (context) |
| \(L\) | Number of layers |
| bytes | Bytes per parameter (e.g. **2** for BF16) |

### 2.1 Attention with Shared KV Heads

Let heads be indexed \(i \in \{1,\ldots,h_q\}\). Map each query head \(i\) to a group \(g(i) \in \{1,\ldots,h_{\mathrm{kv}}\}\) with \(\lceil h_q / h_{\mathrm{kv}} \rceil\) or balanced assignment (typically \(h_q\) divisible by \(h_{\mathrm{kv}}\)).

For token position \(t\), batch element \(b\):

\[
Q_{b,t,i} = X_{b,t} W_Q^{(i)}, \qquad
K_{b,t,g} = X_{b,t} W_K^{(g)}, \qquad
V_{b,t,g} = X_{b,t} W_V^{(g)}.
\]

Attention logits for head \(i\) belonging to group \(g = g(i)\):

\[
S^{(i)}_{t,s} \;=\; \frac{1}{\sqrt{d_k}}\, Q_{b,t,i}\, K_{b,s,g}^{\mathsf T},
\qquad
\alpha^{(i)}_{t,\cdot} = \mathrm{softmax}\bigl(S^{(i)}_{t,\cdot}\bigr),
\qquad
O_{b,t,i} = \sum_s \alpha^{(i)}_{t,s}\, V_{b,s,g}.
\]

**Only** the **query** projections differ across all \(h_q\) heads; **keys and values** repeat for the \(g\) queries in the same group.

### 2.2 Parameter and Activation Tensor Shapes

Ignoring biases, weight counts for one layer’s attention projections scale as:

- **Queries:** \(h_q \cdot d \cdot d_k \sim O(h_q\, d^2 / h_q) = O(d^2)\) when \(d_k = d/h_q\).
- **Keys & values:** \(2\, h_{\mathrm{kv}} \cdot d \cdot d_k\).

Compared to MHA (where \(h_{\mathrm{kv}} = h_q\)), **K and V parameter tensors** shrink by factor \(h_q / h_{\mathrm{kv}}\). The **output projection** \(W_O\) remains comparable to MHA (it maps concatenated heads back to \(d\)).

---

## 3. Memory and KV Cache Formulas

### 3.1 Per-Token KV Bytes

Focus on **cached tensors** for past positions. For one layer, **per sequence**, storing keys and values for all heads that are **materialized** in the cache:

- **MHA:** \(2 \times h_q \times d_k\) scalars per cached timestep (K and V).
- **GQA:** \(2 \times h_{\mathrm{kv}} \times d_k\) scalars per cached timestep.

With **precision** `bytes` bytes per scalar:

\[
\text{KV\_bytes\_per\_token\_per\_layer}
= 2 \cdot h_{\mathrm{kv}} \cdot d_k \cdot \texttt{bytes}.
\]

Across \(L\) layers and context \(T\) **completed** tokens (implementation details vary slightly for current vs cached layout):

\[
\text{KV\_cache} \approx B \cdot T \cdot L \cdot 2 \cdot h_{\mathrm{kv}} \cdot d_k \cdot \texttt{bytes}
\quad\text{(GQA)}.
\]

### 3.2 Reduction Factor \(h_q / h_{\mathrm{kv}}\)

**Ratio** of MHA KV to GQA KV per token (same \(d_k\), precision):

\[
\frac{2 h_q d_k}{2 h_{\mathrm{kv}} d_k} = \frac{h_q}{h_{\mathrm{kv}}}.
\]

For LLaMA‑3 8B: \(32/8 = 4\) — **four times** fewer KV bytes than MHA.

### 3.3 Comparison Table (MHA vs MQA vs GQA)

| Quantity (per layer, schematic) | MHA | GQA (\(h_{\mathrm{kv}}=G\)) | MQA |
|---------------------------------|-----|------------------------------|-----|
| Distinct \(W_K, W_V\) pairs | \(h_q\) | \(G\) | \(1\) |
| KV scalars cached / token (width \(d_k\) each) | \(2 h_q d_k\) | \(2 G d_k\) | \(2 d_k\) |
| Reduction vs MHA (KV footprint) | \(1\times\) | \((h_q/G)\times\) | \(h_q\times\) |
| Diversity of attention keys (bases) | maximal | intermediate | minimal |

**Reading guide:** “Reduction vs MHA” applies to **KV tensor footprint**, not necessarily to total static parameter count (FFN blocks are unchanged).

### 3.4 Decode-Step Attention FLOPs (Informal)

For one new token attending a length-\(T\) prefix, each query head forms dot products with all past keys of its **assigned** KV group: \(\sim 2 T d_k\) multiply-adds to build one row of logits before softmax (constants absorbed). **Forming** \(Q\) still uses all \(h_q\) head projections; **cached** side stores only \(h_{\mathrm{kv}}\) key/value “lanes” across time. Thus:

- **Memory** wins track \(h_q/h_{\mathrm{kv}}\) (what you **store** per timestep).
- **Compute** is architecture- and kernel-dependent: fused attention may still stream full \(Q\) against **replicated or grouped** \(K\) views; do **not** assume FLOPs drop by the same factor as KV bytes without profiling.

---

## 4. Why Grouping Works (Derivation Sketch)

**Informal Hilbert-space view:** Each attention head implements a **low-rank** bilinear map \((Q_i, K_g)\mapsto\) scores and \((\alpha, V_g)\mapsto\) output. Sharing \(K_g, V_g\) across a **block** of queries \(\{Q_{g,1},\ldots,Q_{g,g}\}\) constrains the set of attainable attention maps to those where **keys/values are identical** within the group—**but** each \(Q_{g,j}\) still induces **different softmax weights** \(\alpha_{g,j}\). Thus heads remain **distinguishable** through **routing** even when **reading** the same \(K,V\) content.

**Mean-pooling conversion (training from MHA):** Given group \(\mathcal{G}\) of original MHA key matrices \(\{W_K^{(i)} : i\in\mathcal{G}\}\), the pooled

\[
\bar W_K^{(\mathcal{G})} = \frac{1}{|\mathcal{G}|}\sum_{i\in\mathcal{G}} W_K^{(i)}
\]

minimises \(\sum_{i\in\mathcal{G}} \mathbb{E}_X\bigl[\|X W_K^{(i)} - X W\|_F^2\bigr]\) in \(W\) (same fixed-point intuition as averaging vectors for least-squares). Brief **continued training** retunes the dependent parameters so the reduced-parameter model matches the original **teacher** manifold.

### 4.1 Algebraic View of Shared K, Diverse \(\alpha\)

Fix group \(g\) and timestep \(t\). Let \(q_{g,j}\) denote the \(j\)-th query vector in group \(g\) (there are \(g\) such indices). All use the **same** key matrix row slice \(K_{b,\cdot,g}\). Row-\(t\) attention for head \((g,j)\):

\[
\alpha^{(g,j)}_{t,s} \;=\; \frac{\exp\bigl((q_{g,j}^{\mathsf T} k_{s,g})/\sqrt{d_k}\bigr)}{\sum_{s'} \exp\bigl((q_{g,j}^{\mathsf T} k_{s',g})/\sqrt{d_k}\bigr)}.
\]

Different \(q_{g,j}\) induce **different** categorical distributions over the same support \(\{k_{s,g}\}\); hence **routing** differs even when **read operands** (_keys, values_) are shared. The constraint binds **which bilinear forms** can be realised, not whether heads collapse to identical outputs.

### 4.2 Why Not Always MQA?

With \(h_{\mathrm{kv}}=1\), one \(k_{s}\) must simultaneously align well with **all** \(h_q\) query directions in expectation—**over-constrained** versus data diversity. **GQA** relaxes rank of the shared subspaces (more \(K,V\) bases) while retaining sublinear KV memory.

---

## 5. LLaMA‑3 Setting and Numerical Example

### NUMERICAL EXAMPLES (worked constants)

| Quantity | Formula | Value (stated assumptions) |
|----------|---------|----------------------------|
| Heads per group | \(h_q/h_{\mathrm{kv}}\) | \(32/8 = 4\) |
| Per-head \(d_k\) | \(d_{\text{model}}/h_q\) | \(4096/32 = 128\) |
| KV bytes / token / layer | \(2 h_{\mathrm{kv}} d_k \times\) BF16 | \(2\cdot8\cdot128\cdot2 = 4096\) B |
| vs MHA KV | ratio \(h_q/h_{\mathrm{kv}}\) | **4×** smaller |

**Assumptions:** \(d_{\text{model}} = 4096\), \(h_q = 32\), \(h_{\mathrm{kv}} = 8\), hence \(d_k = d/h_q = 128\), **BF16** (`bytes = 2`), **\(L = 32\)** layers (typical 8B-class stack; adjust for your checkpoint).

**Per token, one layer — KV only (GQA):**

\[
2 \cdot h_{\mathrm{kv}} \cdot d_k \cdot \texttt{bytes}
= 2 \cdot 8 \cdot 128 \cdot 2 = 4096\ \text{bytes} = 4\ \text{KiB}.
\]

**MHA baseline** (same \(d_k\) per head, 32 distinct K/V):

\[
2 \cdot 32 \cdot 128 \cdot 2 = 16{,}384\ \text{bytes} = 16\ \text{KiB}.
\]

**Ratio:** \(16/4 = 4 = h_q / h_{\mathrm{kv}}\).

**Full KV cache (order-of-magnitude):** \(B\) sequences, context \(T\), \(L\) layers:

\[
\text{KV\_GQA} \approx B \cdot T \cdot L \cdot 4\ \text{KiB}
\quad\text{vs}\quad
\text{KV\_MHA} \approx B \cdot T \cdot L \cdot 16\ \text{KiB}
\]

under the above toy accounting (layout/padding can add a few percent).

**Numerical snapshot** (\(B=1\), \(T=4096\), \(L=32\)):

- GQA: \(1 \cdot 4096 \cdot 32 \cdot 4\ \text{KiB} \approx 512\ \text{MiB}\).
- MHA: \(\approx 2\ \text{GiB}\) under the same assumptions.

These figures align in scale with the common **4× rule** at equal \(d_k\) and head design.

---

## 6. Checkpoint Conversion (Mean Pooling)

**Input:** MHA tensors \(W_K^{(1)},\ldots,W_K^{(h_q)}\). Partition into \(h_{\mathrm{kv}}\) contiguous groups of size \(g=h_q/h_{\mathrm{kv}}\). **Output:**

\[
W_K^{[\gamma]} \leftarrow \frac{1}{g}\sum_{j=1}^{g} W_K^{(\,(\gamma-1)g+j\,)}, \qquad
W_V^{[\gamma]} \leftarrow \frac{1}{g}\sum_{j=1}^{g} W_V^{(\,(\gamma-1)g+j\,)}.
\]

Optional **fine-tune** on a small fraction of data recovers perplexity close to training GQA **from scratch**.

---

## 7. Pseudocode

```
# hyperparameters: h_q, h_kv, d_model, layers L
assert h_q % h_kv == 0
g_heads = h_q // h_kv            # queries per KV group
d_k = d_model // h_q             # per-head dim (typical path)

def map_q_head_to_kv_group(q_head_idx):
    return q_head_idx // g_heads   # 0 .. h_kv-1

def forward_attention_step(X, Wq_stack, Wk_g, Wv_g, caches):
    # X: [B, 1, d] single new token; caches store K,V per layer per group
    # Wq_stack: list of h_q matrices; Wk_g, Wv_g: lists length h_kv
    Q = [X @ Wq_stack[i] for i in range(h_q)]
    K_new = [X @ Wk_g[j] for j in range(h_kv)]
    V_new = [X @ Wv_g[j] for j in range(h_kv)]
    append(K_new, V_new) to caches (per-layer structure)
    for each query head i:
        g = map_q_head_to_kv_group(i)
        scores = matmul(Q[i], cache_K[g].transpose) / sqrt(d_k)  # attends all past + self
        attn = softmax(scores, axis=keys)
        out_i = attn @ cache_V[g]
    concat out_1..out_hq, project with W_o
    return output
```

---

## 8. Common Mistakes

- ❌ **Confusing \(d_k = d/h_{\mathrm{kv}}\)** — typically per-head width is \(d/h_q\) (query heads set the split), **not** divided by KV groups. ✓ **Use** \(d_k = d_{\text{model}}/h_q\) unless your codebase documents otherwise.
- ❌ **Counting query tensors in KV cache** — only **\(K,V\)** tensors persist across autoregressive steps. ✓ **Scale** bytes with \(h_{\mathrm{kv}}\), not \(h_q\).
- ❌ **Expecting MQA quality with tiny \(h_{\mathrm{kv}}\)** — extreme sharing hurts. ✓ **GQA** with \(h_{\mathrm{kv}} \approx h_q/4\) is a usual **Pareto** point.
- ❌ **Ignoring layout/padding** — fused kernels group heads differently. ✓ **Profile** actual allocator sizes on target hardware.

---

## 9. Exercises

1. **Algebra:** Show that for fixed \(d_{\text{model}}\), halving \(h_q\) **doubles** \(d_k\) if you maintain \(d = h_q d_k\). Discuss implications for **throughput vs expressivity** (fewer heads, wider per-head spaces).
2. **Deployment:** For LLaMA‑3 8B (\(L=32\), BF16), compute KV cache **GiB** for \(B=16\), \(T=8192\), under GQA \(h_{\mathrm{kv}}=8\); compare to MHA at same \(d_k\) layout.
3. **Conversion:** Write the **least-squares** proof sketch that matrix averaging minimises \(\sum_i \|X W_i - X W\|_F^2\) over \(W\) for fixed data covariance (hint: vectorise + normal equations).
4. **Design:** Argue why **rotary embeddings (RoPE)** on \(Q\) and \(K\) remain coherent when multiple \(Q\) heads share one \(K\) projection—what must be identical per group?

---

*End of note — GQA bridges representation depth and inference economics; always reconcile formulas with your framework’s layout (`n_heads`, `num_kv_heads`, GQA repeat interleave).*
