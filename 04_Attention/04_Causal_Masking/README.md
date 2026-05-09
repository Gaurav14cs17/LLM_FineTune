# Understanding Causal (Autoregressive) Masking

*Future-blind attention for decoder-only LMs: masks, softmax mechanics, teacher forcing, and prefix-LM hybrids*

---

**Causal masking** forces each position in a sequence to attend only to **itself and previous** tokens. Without it, a Transformer decoder could “peek” at future tokens—making training **too easy** and breaking **alignment** with autoregressive inference, where the future does not exist yet. The standard implementation adds a **negative infinity** (or large negative) mask on disallowed positions **before** softmax, zeroing those probabilities exactly.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why the Future Must Be Masked](#11-why-the-future-must-be-masked)
   - [1.2 The Causal Mask Matrix](#12-the-causal-mask-matrix)
2. [Interaction with Softmax](#2-interaction-with-softmax)
   - [2.1 From Logits to Probabilities](#21-from-logits-to-probabilities)
   - [2.2 Mask = Additive Logit Penalty](#22-mask--additive-logit-penalty)
   - [2.3 Equivalent Characterisations of Causal Attention](#23-equivalent-characterisations-of-causal-attention)
   - [2.4 On nan Risks](#24-on-nan-risks)
3. [Teacher Forcing and Parallel Training](#3-teacher-forcing-and-parallel-training)
4. [Prefix Language Models (Bidirectional Prompt + Causal Gen)](#4-prefix-language-models-bidirectional-prompt--causal-gen)
5. [Connection to Training Parallelism](#5-connection-to-training-parallelism)
   - [5.1 Tensor-Parallel and Activation Sharding (Pointer)](#51-tensor-parallel-and-activation-sharding-pointer)
6. [Numerical Example](#6-numerical-example)
   - [6.1 Larger Worked Row (L = 5, Stable Probs)](#61-larger-worked-row-l--5-stable-probs)
7. [Pseudocode](#7-pseudocode)
8. [Common Mistakes](#8-common-mistakes)
9. [Exercises](#9-exercises)

---

## 1. Overview

### VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(L\) | Sequence length (positions \(0\ldots L-1\)) |
| \(S \in \mathbb{R}^{L\times L}\) | Raw attention score matrix (pre-mask, pre-softmax per head) |
| \(M \in \{0, -\infty\}^{L\times L}\) | Causal mask: \(M_{ij}=0\) if \(j\le i\), else \(-\infty\) (implementations use a large finite negative) |
| \(\tilde{S}\) | Masked scores \(\tilde{S} = S + M\) |
| \(A\) | Attention weights \(A = \mathrm{softmax}(\tilde{S})\) (row-wise) |
| \(Q,K,V\) | Query/key/value tensors for one head |

### ┌─────────────────────────────────────────────────────────────┐
### │ INTUITION: read-only past, write-at-now                     │
### └─────────────────────────────────────────────────────────────┘

```
  time ─────────────────────────────────────────►
         t=0   t=1   t=2   t=3   t=4
        ┌─────┬─────┬─────┬─────┬─────┐
   t=0  │  ✓  │  ?  │  ?  │  ?  │  ?  │   "?" = FUTURE (must NOT see)
   t=1  │  ✓  │  ✓  │  ?  │  ?  │  ?  │
   t=2  │  ✓  │  ✓  │  ✓  │  ?  │  ?  │   Allowed region = lower triangle
   t=3  │  ✓  │  ✓  │  ✓  │  ✓  │  ?  │
   t=4  │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │
        └─────┴─────┴─────┴─────┴─────┘
```

At row \(i\) (query position \(i\)), only columns \(j\le i\) may carry nonzero mass after softmax.

### 1.1 Why the Future Must Be Masked

- **Training:** If position \(i\) could attend \(j>i\), its representation absorbs **token \(j\)**’s identity. The next-token loss at \(i\) then exploits **information leakage**; the model need not **predict** those future symbols from the prefix—it can copy from the “cheat sheet” inside attention.
- **Inference:** When generating left-to-right, **tokens beyond \(i\)** are undefined. A model trained with future access learns a **different conditional** than the one used at test time (**exposure bias** mismatch is still handled partly by teacher forcing, but **mask removal** would be an outright train/serve inconsistency).

### 1.2 The Causal Mask Matrix

For length \(L\), ideal mask entries (additive to logits):

\[
M_{ij} = \begin{cases}
0, & j \le i, \\
-\infty, & j > i.
\end{cases}
\]

**Lower-triangular allowed region** includes the diagonal (self-attention). Upper triangle is **blocked**.

**Example \(L=5\)** (symbols `0` = keep, `-∞` = block):

```
        j=0    1     2     3     4
i=0   [  0   -∞    -∞    -∞    -∞ ]
i=1   [  0    0    -∞    -∞    -∞ ]
i=2   [  0    0     0    -∞    -∞ ]
i=3   [  0    0     0     0    -∞ ]
i=4   [  0    0     0     0     0 ]
```

---

## 2. Interaction with Softmax

### 2.1 From Logits to Probabilities

Row \(i\) of attention (single head, pre-mask scores \(s_{ij}\)):

\[
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^{L} \exp(s_{ik})}.
\]

Properties: \(\alpha_{ij} \ge 0\), \(\sum_j \alpha_{ij} = 1\). **Every** position starts with **positive** mass unless pre-softmax scores are \(-\infty\).

### 2.2 Mask = Additive Logit Penalty

Define masked logits \(\tilde{s}_{ij} = s_{ij} + M_{ij}\). For **blocked** \(j>i\):

\[
\tilde{s}_{ij} = -\infty \quad\Rightarrow\quad \exp(\tilde{s}_{ij}) = 0 \quad\Rightarrow\quad \alpha_{ij} = 0.
\]

The remaining softmax **renormalises** over \(j \le i\) only:

\[
\alpha_{ij} = \frac{\exp(s_{ij}) \cdot \mathbf{1}_{j \le i}}{\sum_{k \le i} \exp(s_{ik})}, \qquad j \le i,
\]

which is exactly **causal normalised attention** over the permitted past.

### NUMERICAL ILLUSTRATION (tiny \(L=4\))

Suppose row \(i=1\) raw scores (before mask) are \([s_{10}, s_{11}, s_{12}, s_{13}] = [1.0,\; 2.0,\; 0.5,\; 3.0]\). Future columns **should not count**. Masked row:

\[
\tilde{s}_{1\cdot} = [1.0,\; 2.0,\; -\infty,\; -\infty].
\]

Numerically,

\[
\exp(\tilde{s}) \approx [e^1,\, e^2,\, 0,\, 0],\quad
Z = e^1 + e^2,\quad
\alpha_{10} = \frac{e^1}{Z},\; \alpha_{11} = \frac{e^2}{Z},\; \alpha_{12}=\alpha_{13}=0.
\]

### 2.3 Equivalent Characterisations of Causal Attention

Let \(\mathcal{A}_i = \{0,1,\ldots,i\}\) denote admissible key indices for query \(i\). Masking **before** softmax yields weights

\[
\alpha_{ij} = \begin{cases}\displaystyle
\frac{\exp(s_{ij})}{\sum_{k \in \mathcal{A}_i} \exp(s_{ik})}, & j \in \mathcal{A}_i,\\
0, & j \notin \mathcal{A}_i.
\end{cases}
\]

This is identical to **projected softmax** onto coordinate subsets—what the ad hoc “set logits to \(-\infty\)” implements **exactly**, not approximately, **under exact arithmetic** (finite precision uses a big negative instead of \(-\infty\)).

### 2.4 On \(\mathrm{nan}\) Risks

If **all** admissible logits in a row are \(-\infty\) (e.g., degenerate padding mask), \(\exp\) sums to **zero** and softmax **divides by zero**. Libraries return **uniform** or **NaN** depending on safeguards—always ensure **each row** has **at least one** permitted key (decoder usually has the **diagonal** self-token, which fixes this).

---

## 3. Teacher Forcing and Parallel Training

**Goal:** predict token \(x_t\) from prefix \(x_{<t}\) with distribution \(p_\theta(x_t \mid x_{<t})\).

**Parallel trick:** feed the **entire** sequence in one forward pass, but **mask** attention so position \(t\) only depends on \(x_{\le t}\). The per-position losses

\[
\mathcal{L} = -\sum_{t=1}^{L} \log p_\theta(x_t \mid x_{<t})
\]

are **independent** across \(t\) **given** the masked attention window, so they can be summed in one graph build.

```
PREFIX CONSISTENCY
──────────────────
Token positions:  x0   x1   x2   x3
Loss at t:       p(x1|x0) p(x2|x0,x1) p(x3|x0,x1,x2) ...
Attention row t:  sees ≤t          sees ≤t          sees ≤t
```

This is **teacher forcing** (feed ground-truth \(x_{<t}\) as inputs). Without causal mask, row \(t\) could also see **ground-truth future** \(x_{>t}\), collapsing the learning signal.

---

## 4. Prefix Language Models (Bidirectional Prompt + Causal Gen)

**Use case:** a prompt is **known in full** before generation; subsequent tokens remain **autoregressive**.

Construct a **segment-wise** mask \(M^\*\):

- For pairs \((i,j)\) **both in the prompt region** \(\mathcal{P}\), allow **bidirectional** attention (often **full** or local window inside \(\mathcal{P}\)).
- For positions in the **generated** region, enforce **strict causality** among generated timesteps **and** allow attending to **all prompt keys** (prefix cache).

Schematically for prompt indices \(\{0,\ldots,P-1\}\) and continuation \(\{P,\ldots,L-1\}\):

\[
M^\*_{ij} =
\begin{cases}
0, & i,j \in \mathcal{P}\ \text{(many variants: full or block-local)},\\
0, & i \ge P,\ j \le i \ \text{(causal over full prefix+gen so far)},\\
-\infty, & \text{otherwise forbidden pairs (e.g., } j>i \text{ within gen)}.
\end{cases}
\]

**Implementation note:** frameworks often realise this by **setting separate attention blocks** (encoder–decoder style) or **fused masks**; always check that **generated \(i\)** cannot attend **future generated \(j>i\)** while **still** reading prompt \(k<P\).

---

## 5. Connection to Training Parallelism

**Why masking unlocks speed:** A stack of \(L\) autoregressive RNN steps is inherently **sequential** in time. A masked Transformer computes **all positions’** representations **in parallel** on modern accelerators, because **dependencies** are only **backward in time** within the mask’s lower envelope.

**Complexity sketch (per layer):** Attention with causal mask still forms an \(L\times L\) score matrix but the **upper triangle is dead** after softmax—numerically zeros. Some kernels **avoid writing** those entries (FlashAttention-style) by knowledge of causality, shrinking memory traffic. Even without such kernels, **correctness** demands masking during **softmax**, not only after.

**Microbatching / sequence packing:** Causal masks extend to **padded mini-batches** by adding **padding masks** \(+\) **causal** masks (typically combine additively in logits with further \(-\infty\) on pad keys).

### 5.1 Tensor-Parallel and Activation Sharding (Pointer)

When sequences are **sharded** across devices, the causal property must hold **post-sharding**. In practice, attention kernels attach a **global position id** per token so causality is defined in **original sequence order**, not local shard order.

---

## 6. Numerical Example

**Setup:** Single head, \(d_k=1\) for simplicity (so \(s_{ij} = q_i k_j\) reduces to scalars). Take \(L=3\) and assume rows before mask:

Row 0: \([s_{00}, s_{01}, s_{02}] = [0, 1, 2]\)  
Row 1: \([0, 0, 1]\)  
Row 2: \([1, 1, 1]\)

**Mask \(M\) causes rows:**

Row 0 masked: \([0,-\infty,-\infty]\Rightarrow \alpha_{00}=1\).  
Row 1 masked: \([0,0,-\infty]\Rightarrow \mathrm{softmax}([0,0]) = [0.5,0.5]\).  
Row 2 masked: \([1,1,1]\Rightarrow \mathrm{softmax} = [1/3,1/3,1/3]\).

**Interpretation:** Even if raw scores favour futures, the mask **removes** their mass; row 0 cannot attend forward; row 1 cannot see index 2.

### 6.1 Larger Worked Row (\(L=5\), Stable Probs)

Take \(i=4\) (full history visible). Suppose raw logits are

\[
s_{4\cdot} = [-1,\; 0,\; 1,\; 2,\; 3].
\]

**No mask:** softmax gives positive mass on all five positions.  
**Causal:** no additional masking needed for row \(i=4\) (all \(j\le 4\) allowed)—this row equals full attention over the prefix.

Contrast **row \(i=1\)** with the same **raw** scores restricted to first two columns:

\[
\tilde{s}_{1\cdot} = [-1,\; 0,\; -\infty,\; -\infty,\; -\infty].
\]

\[
\alpha_{10} = \frac{e^{-1}}{e^{-1}+e^{0}} = \frac{1}{1+e},\quad
\alpha_{11} = \frac{e^{0}}{e^{-1}+e^{0}} = \frac{e}{1+e},
\]

and zeros elsewhere—**mass redistributes** entirely within the prefix.

---

## 7. Pseudocode

```
# s: pre-softmax attention scores, shape [L, L] for one head (queries row-wise)
# returns attention weights A with zeros on forbidden positions

def causal_mask_matrix(L, neg=-1e9):  # finite stand-in for -inf in float32
    M = neg * (1 - tril(ones(L,L)))    # 0 on lower triangle incl. diag, neg upper
    return M

def apply_causal_attention(scores_LxL, neg=-1e9):
    L = scores_LxL.shape[0]
    S_tilde = scores_LxL + causal_mask_matrix(L, neg)
    # row softmax
    expS = exp(S_tilde)                 # exp(-inf) == 0 in stable lib
    return expS / sum(expS, axis=-1, keepdims=True)

# Training loss with teacher forcing, decoder-only
# logits[t] predicts x[t+1]; build scores with causal mask so pos t never sees x[t+1]
```

**Stability tip:** Use **masked_fill** before softmax rather than multiplying after—post-softmax zeroing **does not** fix partition function misuse in naive softmax.

---

## 8. Common Mistakes

- ❌ **Softmax first, multiply by 0 mask** — wrong partition: you still normalised over **future** keys. ✓ **Add** large negative to logits **before** softmax, or use **masked_softmax** that recomputes the denominator on allowed entries.
- ❌ **Using `-inf` in low-precision (FP16/BF16)** without care — overflows/NaNs. ✓ Use **large finite** negative (\(\approx -10^4\)) consistent with dtype.
- ❌ **Bidirectional encoder mask in decoder** — leaks answers during LM training. ✓ **Causal** for decoder; **full** only for pure encoding of **known** contexts with appropriate objective.
- ❌ **Padding forgotten** — model attends to `<pad>` keys. ✓ Combine **padding mask** with **causal** mask on keys (and sometimes queries).
- ❌ **Prefix LM: full bidirectional among generated tokens** — leakage within continuation. ✓ Keep **causal** among generated steps; **prompt** rules per design.

---

## 9. Exercises

1. **Write** the explicit \(4\times 4\) matrix \(M\) with \(0\) / \(-\infty\) entries (you may symbolic \(-\infty\)). Verify \(\mathrm{softmax}(s+M)\) upper triangle is **exactly zero** for random finite \(s\).
2. **Proof sketch:** show that for any row \(i\), if \(\tilde{s}_{ij}=-\infty\) for all \(j\in J\), those indices contribute **zero** mass and softmax equals softmax on the complement **renormalised**.
3. **Prefix LM:** draw a **\(6\times 6\)** allowed-score pattern where positions \(0\ldots 2\) are prompt and \(3\ldots 5\) are generated with **full prompt bidirectional** but **causal** generation; justify each forbidden cell.
4. **Parallelism:** explain how `torch.compile` / fused attention can exploit **causality** to skip materialising upper triangles—contrast memory bytes naive vs Flash-style **without** changing math.

---

*Causal masking is the algebraic gatekeeper of autoregressive Transformers: it makes the factorisation \(p(x)=\prod_t p(x_t\mid x_{<t})\) **honest** in both training and serving.*
