# Feed-Forward Network with SwiGLU

The Transformer mixes information across positions with attention; the **position-wise feed-forward network (FFN)** is where most parameters live and where per-token nonlinear **capacity** is concentrated. This note compares the classical GELU/ReLU MLP to **SwiGLU** (Swish-activated Gated Linear Unit), links the **8/3×** width rule to parameter and FLOP budgets, and contrasts activations used in modern LLMs.

---

## Table of Contents

- [INTUITION](#intuition)
- [VARIABLES & SHAPES](#variables--shapes)
- [Standard Two-Matrix FFN](#standard-two-matrix-ffn)
- [SwiGLU: Definition and Geometry](#swiglu-definition-and-geometry)
- [SiLU / Swish: Smooth Gating](#silu--swish-smooth-gating)
- [Parameter Budget: The \(8d/3\) Rule](#parameter-budget-the-8d3-rule)
- [FLOPs Accounting (Per Token)](#flops-accounting-per-token)
- [LLaMA-3 8B: \(d_{\text{ff}} = 14336\)](#llama-3-8b-d_textff--14336)
- [ReLU vs GELU vs SwiGLU (Qualitative)](#relu-vs-gelu-vs-swiglu-qualitative)
- [Key–Value Memory View](#keyvalue-memory-view)
- [NUMERICAL EXAMPLES](#numerical-examples)
- [Pseudocode](#pseudocode)
- [Backward Pass Sketch](#backward-pass-sketch)
- [COMMON MISTAKES](#common-mistakes)
- [EXERCISES](#exercises)
- [REFERENCES](#references)

---

## INTUITION

```
    TOKEN x (d-dim)  ─────┬──────────────────────────────────────────────►
                          │
              ┌───────────┴───────────┐
              │   ATTENTION (mixes)    │   FFN (same fn each row: "recall + refit")
              └───────────┬───────────┘
                          │
     Standard FFN:         │         SwiGLU FFN:
     h = σ(W₁x)            │         a = W_up x
     y = W₂h               │         b = W_gate x
                          │         h = SiLU(b) ⊙ a    ←── gate picks sub-experts
                          │         y = W_down h
                          ▼
              High-rank "pattern → value" map; most MODEL params sit here

        ┌───────────────────────────────────────────────────────────────┐
        │  SiLU(b) ⊙ a  =  smooth, data-dependent routing of channels   │
        │                 (unlike ReLU's hard half-space cut)            │
        └───────────────────────────────────────────────────────────────┘
```

**Roadmap:** a classical FFN is two matmuls with one nonlinearity. SwiGLU **duplicates** the up-projection to obtain a **gate** and a **value** path, then multiplies them—adding **adaptive per-channel curvature** at the cost of a third weight matrix. Shrink hidden width by **\(8/(3\times 4)=\)** **\(2/3\)** relative to a “\(4d\)” MLP (i.e. \(d_{\text{ff}} \approx 8d/3\)) to **preserve** parameter and FLOP parity; production models (e.g. LLaMA-3) often go **wider** (e.g. **\(3.5d\)**) deliberately trading compute for quality.

---

## VARIABLES & SHAPES

| Symbol | Shape / role |
|--------|----------------|
| \(d\) | Model **hidden size** (residual stream width); e.g. 4096 in LLaMA-3 8B configs cited in public cards |
| \(d_{\text{ff}}\) | FFN inner (“bottleneck expansion”) dimension |
| \(x\) | Single-token input in \(\mathbb{R}^d\) (row of \(X\in\mathbb{R}^{T\times d}\)) |
| \(W_1, W_2\) | Standard FFN: \(W_1\in\mathbb{R}^{d\times d_{\text{ff}}}\), \(W_2\in\mathbb{R}^{d_{\text{ff}}\times d}\) |
| \(W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}\) | SwiGLU: \(W_{\text{up}}, W_{\text{gate}}\in\mathbb{R}^{d\times d_{\text{ff}}}\), \(W_{\text{down}}\in\mathbb{R}^{d_{\text{ff}}\times d}\) |
| \(\sigma\) | Logistic sigmoid \(\sigma(z) = 1/(1+e^{-z})\) |
| \(\mathrm{SiLU}(z)\) | \(z\cdot \sigma(z)\) (a.k.a. Swish with \(\beta=1\)) |
| \(\odot\) | Elementwise (Hadamard) product of two length-\(d_{\text{ff}}\) vectors |

Bias vectors are often omitted in large LMs (especially with RMSNorm); including them adds \(O(d_{\text{ff}})\) parameters, negligible next to weights.

---

## Standard Two-Matrix FFN

For one token (column-oriented convention \(x \in \mathbb{R}^{1\times d}\)):

\[
h \;=\; \varphi(x W_1), \qquad y \;=\; h W_2
\]

Common choices: \(\varphi=\) **ReLU**, **GELU**, **GeLU-approx**. Shapes: \(h \in \mathbb{R}^{1\times d_{\text{ff}}}\).

**Parameter count (no bias):**

\[
|\theta|_{\text{std}} \;=\; d d_{\text{ff}} + d_{\text{ff}} d \;=\; 2 d d_{\text{ff}}.
\]

The classic “\(4d\)” MLP sets \(d_{\text{ff}} = 4d\), giving **\(8d^2\)** parameters in the block—until SwiGLU, this was the usual baseline for scaling discussions.

**Universality intuition:** width \(d_{\text{ff}}\) trades off **expressivity** (approximating functions of a single token’s vector) versus memory/compute. Research on “key-value” views of FFNs interprets columns of \(W_1\) / rows of \(W_2\) as soft memory entries; SwiGLU adds a **per-entry gate** on how strongly each “value” contributes.

---

## SwiGLU: Definition and Geometry

**Exact modern form (matches Shazeer; used in PaLM/LLaMA line):**

\[
a \;=\; x W_{\text{up}}, \qquad b \;=\; x W_{\text{gate}}
\]
\[
h \;=\; \mathrm{SiLU}(b) \odot a \;=\; (b \odot \sigma(b)) \odot a
\]
\[
y \;=\; h W_{\text{down}}
\]

Some texts swap labels \(W_1/W_3\); always identify **two up-projections** and **one down**. Gating **multiplies** the value path by a **smooth, input-dependent** vector in \((0,\infty)\) (asymptotically like **scale** per channel).

**Why gating helps:** \(\mathrm{SiLU}(b_i)\) creates **sparse-ish but smooth** modulation of \(a_i\). Large negative \(b_i\) suppresses \(a_i\) without a hard zero (contrast ReLU on \(a\) alone); large positive \(b_i\) lets \(a_i\) pass. Across the batch, different tokens activate different subsets of channels—**adaptive computation** in width without explicit routing layers.

---

## SiLU / Swish: Smooth Gating

**Definition:**

\[
\mathrm{SiLU}(z) = z \cdot \sigma(z).
\]

**Derivatives (chain rule for training):**

\[
\frac{\mathrm{d}}{\mathrm{d}z}\mathrm{SiLU}(z)
= \sigma(z) + z\,\sigma(z)(1-\sigma(z))
= \sigma(z)\,\bigl(1 + z(1-\sigma(z))\bigr).
\]

Smooth nonlinearity avoids **ReLU dead zones** on the gate path; unlike GELU on the **value** path only, the **product** SiLU\(\odot\) allows **multiplicative** interactions **before** the down-projection—empirically strong for LM quality at scale.

---

## Parameter Budget: The \(8d/3\) Rule

SwiGLU uses **three** matrices of shapes \(d\times d_{\text{ff}}\) (up and gate) and \(d_{\text{ff}}\times d\) (down):

\[
|\theta|_{\text{SwiGLU}} \;=\; 3 d d_{\text{ff}}.
\]

Match the parameter count of a standard MLP with inner width \(d_{\text{ff}}^{\text{(std)}} = 4d\):

\[
3 d d_{\text{ff}}^{\text{(swi)}} \;=\; 2 d \cdot (4d)
\quad\Rightarrow\quad
d_{\text{ff}}^{\text{(swi)}} \;=\; \frac{8}{3}\,d.
\]

So **replacing** a ReLU/GELU \(4d\) FFN with SwiGLU **without** increasing total footprint suggests **shrinking** inner width from **\(4d\)** to **\(8d/3\)**.

---

## FLOPs Accounting (Per Token)

A common multiply-add count for a matrix product of shapes **(1,d)** times **(d, d\_{\text{ff}})** is **\(2\,d\,d_{\text{ff}}\)** FLOPs (same for **(d\_{\text{ff}},d)** on the backward or transposed).

**Standard FFN (forward, one token):**

\[
\text{FLOPs}_{\text{std}} \;=\; 2 d d_{\text{ff}}^{\text{(std)}} + 2 d_{\text{ff}}^{\text{(std)}} d
\;=\; 4 d d_{\text{ff}}^{\text{(std)}}.
\]

**SwiGLU:**

\[
\text{FLOPs}_{\text{Swi}} \;=\; 2 d d_{\text{ff}} + 2 d d_{\text{ff}} + 2 d d_{\text{ff}}
\;=\; 6 d d_{\text{ff}}.
\]

Set \(6 d d_{\text{ff}}^{\text{(swi)}} = 4 d d_{\text{ff}}^{\text{(std)}}\) with \(d_{\text{ff}}^{\text{(std)}}=4d\):

\[
d_{\text{ff}}^{\text{(swi)}} \;=\; \frac{4}{6}\cdot 4d \;=\; \frac{2}{3}\cdot 4d \;=\; \frac{8d}{3}.
\]

So the **same** **8/3** inner width balances **both** parameters **and** this forward **FLOP** tally against the canonical \(4d\) two-matrix baseline.

---

## LLaMA-3 8B: \(d_{\text{ff}} = 14336\)

Public architecture cards often list **\(d=4096\)** and **\(d_{\text{ff}}=14336\)** for LLaMA-3 8B-class stacks (exact recipe varies by tier). Note:

\[
\frac{14336}{4096} = 3.5
\quad\text{(i.e. \(d_{\text{ff}} = 3.5\,d\)).}
\]

This is **wider** than \(8/3 \approx 2.67\)—a deliberate **over-capacity** choice: more FFN FLOPs and parameters per layer vs the “matched-to-4d-baseline” recipe, **buying accuracy** (especially factual/extractive behavior that correlates with MLP capacity).

**Illustrative per-layer FFN parameter slice (no bias, counts match \(\mathbb{R}^{d\times d_{\text{ff}}}\) layouts):**

\[
3 \times 4096 \times 14336 \;\approx\; 1.76\times 10^8 \ \text{params per layer.}
\]

Across \(L\) layers, FFN dominates total parameter mass in dense decoder LMs (often on the order of 60–75% of weights depending on \(L\), \(d\), and attention sharding).

---

## ReLU vs GELU vs SwiGLU (Qualitative)

| Activation style | Behavior | Typical LLM-era notes |
|------------------|----------|------------------------|
| **ReLU** | Hard zero for \(z<0\); cheap | Dead units; piecewise linear → worse LM scaling vs smooth gates |
| **GELU** | Smooth, probabilistic mix | Strong baseline in BERT/GPT eras; no multiplicative gate path |
| **SwiGLU** | Piecewise **product** structure | Often **best quality** at equal budget among MLP variants when width tuned |

Empirically (large-scale LM pretraining), ordering is commonly reported as **SwiGLU > GELU > ReLU**, with SwiGLU’s gain tied to **gated feature routing** rather than smoothness alone.

---

## Key–Value Memory View

Ignoring biases, expand the down-projection row-wise:

\[
y \;=\; \sum_{i=1}^{d_{\text{ff}}} h_i\, W_{\text{down}}[i,:]
\;=\; \sum_{i=1}^{d_{\text{ff}}}
\left(\mathrm{SiLU}(b_i)\cdot a_i\right)\, W_{\text{down}}[i,:].
\]

Each inner unit \(i\) contributes a **rank-one** update to \(y\) scaled by a **gated score**. Interpret \(W_{\text{down}}[i,:]\) as **value vectors** and the gate/value interaction as **address-dependent** read strength—conceptually closer to a **content-addressed** collection of static patterns than a single fixed activation curve.

---

## NUMERICAL EXAMPLES

**Inner width comparison** for \(d=4096\):

- Baseline \(4d\) standard MLP: \(d_{\text{ff}}=16384\), params \(2\cdot 4096\cdot 16384 \approx 1.34\times 10^8\).
- Matched SwiGLU width \(8d/3\): \(d_{\text{ff}} \approx 10922.7\) (rounded in practice to hardware multiples) → same \(\sim 1.34\times 10^8\) param budget when equality holds exactly.

**LLaMA-3 style width \(14336\):**

\[
3\cdot 4096 \cdot 14336 \approx 1.76\times 10^8\ \text{params},
\quad
\text{FLOPs}_{\text forward}\approx 6\cdot 4096 \cdot 14336 \approx 3.52\times 10^8.
\]

**Gating magnitude:** if \(b_i=-2\), \(\sigma(b_i)\approx 0.12\), \(\mathrm{SiLU}(b_i)\approx -0.24\) (small negative gate allows “soft veto” of opposing-signed \(a_i\)); if \(b_i=2\), \(\mathrm{SiLU}(b_i)\approx 1.76\).

---

## Pseudocode

```
function silu(z):
    return z * sigmoid(z)

def ffn_standard(x, W1, W2, phi):        # x: [batch, d]
    h = phi(matmul(x, W1))               # [batch, d_ff]
    return matmul(h, W2)                 # [batch, d]

def ffn_swiglu(x, W_up, W_gate, W_down):
    a = matmul(x, W_up)                  # value path
    b = matmul(x, W_gate)                # gate logits
    h = silu(b) * a                      # elementwise combine
    return matmul(h, W_down)

# Width recipe for "match 4d baseline" SwiGLU (no bias):
def matched_ffn_width(d: int) -> int:
    return round_to_multiple(8 * d / 3, multiplicity=64)  # e.g. TP/alignment
```

---

## Backward Pass Sketch

Let \(h_i = \mathrm{SiLU}(b_i)\,a_i\) with \(a = x W_{\text{up}}\), \(b = x W_{\text{gate}}\). Then

\[
\frac{\partial h_i}{\partial a_i} = \mathrm{SiLU}(b_i),
\qquad
\frac{\partial h_i}{\partial b_i} = a_i \cdot \sigma(b_i)\bigl(1 + b_i(1-\sigma(b_i))\bigr).
\]

Backprop through \(y = h W_{\text{down}}\) routes \(\partial L/\partial h\) into **both** upstream matrices:

\[
\frac{\partial L}{\partial W_{\text{up}}}
\;=\; x^\top \left(\frac{\partial L}{\partial h} \odot \mathrm{SiLU}(b)\right),
\qquad
\frac{\partial L}{\partial W_{\text{gate}}}
\;=\; x^\top \left(\frac{\partial L}{\partial h} \odot a \odot \mathrm{SiLU}'(b)\right).
\]

The gate path carries gradients even when \(a_i\) is small—**unlike** ReLU-gated variants that hard-stop negative pre-activations on the **value** path. This smoothed coupling is part of why optimization at scale is tractable.

---

## COMMON MISTAKES

- ❌ Claim SwiGLU needs **\(4d\)** inner width “like GPT-2” **without** adjusting for the **third** matrix.  
  ✓ Either set \(d_{\text{ff}} \approx 8d/3\) for parity, or **own** the higher compute of wider stacks.

- ❌ Confuse **SiLU gate** with **sigmoid-only GLU** \(a\odot \sigma(b)\). Different gating curvature; LLaMA uses **SiLU** on gate as in **SwiGLU**.  

- ❌ Forget **layout & fused kernels**: math counts param tensors as \(3 d d_{\text{ff}}\); **implementation** may pack \(W_{\text{gate}}\) and \(W_{\text{up}}\) for bandwidth.

- ❌ Assume “\(3.5d\) expansion” is **mandatory** for quality—it's a **model-line** choice; smaller widths still train strong models with different \(L,d,N\) trade-offs.

- ❌ Compare activations at **fixed random init**; ranking emerges after **large-scale** training—not from one forward pass.

---

## EXERCISES

1. **Parameters:** Derive \(d_{\text{ff}} = 8d/3\) by equating \(3 d d_{\text{ff}}\) to \(2 d \cdot 4d\). Plug \(d=2048\); give a rounded multiple-of-128 width.

2. **FLOPs ratio:** For general \(d_{\text{ff}}\), show \(\text{FLOPs}_{\text{Swi}}/\text{FLOPs}_{\text{std}} = 1.5\) when both use the **same** \(d_{\text{ff}}\). Then show equality when \(d_{\text{ff}}^{\text{(Swi)}} = \tfrac{2}{3} d_{\text{ff}}^{\text{(std)}}\).

3. **Derivatives:** For scalar \(h_i=\mathrm{SiLU}(b_i)\cdot a_i\), write \(\partial h_i/\partial a_i\) and \(\partial h_i/\partial b_i\).

4. **Memory capacity (rough):** If each of \(d_{\text{ff}}\) units stores one \(d\)-dim value row in \(W_{\text{down}}\), how many distinct value vectors sit in one layer at \(d_{\text{ff}}=14336\)? How many across **32** layers (order-of-magnitude **integer** count, ignoring interference)?

5. **Quality vs budget:** Explain when a team might pick \(d_{\text{ff}}=3.5d\) over \(8d/3\) **despite** higher FLOPs per token.

---

## REFERENCES

- Noam Shazeer, *GLU Variants Improve Transformer* (2020).  
- LLaMA / LLaMA 3 technical reports (Meta) for reported widths and norms.  
- Geva et al. on key-value / factual structure in FFN layers (interpretability).
