# Residual and Skip Connections in Transformers

*Gradient highways, signal propagation, and pre-norm placement in deep LLM stacks*

---

## Table of Contents

1. [Overview](#1-overview)
2. [INTUITION: The Identity Highway](#2-intuition-the-identity-highway)
3. [Formal Block and Jacobians](#3-formal-block-and-jacobians)
4. [VARIABLES](#4-variables)
5. [Gradient Flow Through One Residual](#5-gradient-flow-through-one-residual)
6. [Chained Residuals and Vanishing Gradients](#6-chained-residuals-and-vanishing-gradients)
7. [Signal Propagation and Depth Scaling](#7-signal-propagation-and-depth-scaling)
8. [Pre-Norm vs Post-Norm](#8-pre-norm-vs-post-norm)
9. [Variance Growth and ResNets](#9-variance-growth-and-resnets)
   - [9.1 Telescope Intuition](#91-intuition-telescope-on-the-forward-pass-not-the-jacobian-product)
   - [9.2 Backward Through L Blocks](#92-backward-through-l-blocks--structure-qualitative)
   - [9.3 When Gradients Can Still Attenuate](#93-when-gradients-can-still-attenuate)
10. [NUMERICAL EXAMPLES](#10-numerical-examples)
11. [Pseudocode](#11-pseudocode)
12. [COMMON MISTAKES](#12-common-mistakes)
13. [EXERCISES](#13-exercises)

---

## 1. Overview

A **residual connection** adds the subnetwork output to its input:

\[
\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}).
\]

In autoregressive Transformers (GPT/LLaMA style), \(\mathcal{F}\) is attention or the feed-forward block (often wrapped in normalisation). The skip path carries a **copy of the signal** that bypasses \(\mathcal{F}\). During backpropagation, that path contributes an **additive identity term** to the Jacobian, which stabilises optimisation in stacks of 32–128 layers.

This note derives \(\partial \mathcal{L}/\partial \mathbf{x}\), compares **pre-norm** and **post-norm**, sketches **signal propagation** at initialisation, and links to **ResNet**-style residual learning.

---

## 2. INTUITION: The Identity Highway

```
┌────────────────────────────────────────────────────────────────────┐
│  RESIDUAL = MAIN ROAD + SIDE ROAD                                   │
│                                                                     │
│              x ────────────────┬──────────────►  +  ──► y = x+F(x) │
│                                │                    ▲                │
│                                │                    │                │
│                                └──►  F(x)  ────────┘                │
│                                                                     │
│     Backward:  ∂L/∂x gets TWO paths:                                │
│       (A) ∂L/∂y  ·  I          ← straight through x                 │
│       (B) ∂L/∂y  ·  ∂F/∂x     ← through the module                │
│                                                                     │
│     Even if ∂F/∂x ≈ 0 early on, (A) alone carries ∂L/∂y backward.   │
└────────────────────────────────────────────────────────────────────┘

WITHOUT skip (deep chain):  ∂L/∂x ∝ ∏ Jacobian  →  can shrink like ρ^L
WITH skip:                  ∂L/∂x always has a term ∂L/∂y  (not multiplied
                            by an unbounded chain of ρ’s for that path)
```

---

## 3. Formal Block and Jacobians

Let \(\mathcal{F}: \mathbb{R}^d \to \mathbb{R}^d\) be differentiable. Define

\[
\mathcal{R}(\mathbf{x}) = \mathbf{x} + \mathcal{F}(\mathbf{x}).
\]

**Forward Jacobian** (total derivative of output w.r.t. input):

\[
\mathbf{J}_{\mathcal{R}}(\mathbf{x})
= \frac{\partial \mathcal{R}}{\partial \mathbf{x}}
= \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \in \mathbb{R}^{d \times d}.
\]

Let \(\mathcal{L}\) be a scalar loss, \(\mathbf{y} = \mathcal{R}(\mathbf{x})\). By the chain rule,

\[
\frac{\partial \mathcal{L}}{\partial \mathbf{x}}
= \left(\frac{\partial \mathcal{L}}{\partial \mathbf{y}}\right)^{\!\top}
\left(\mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right),
\]

or in transposed/VJP form (as implemented in autodiff):

\[
\nabla_{\mathbf{x}} \mathcal{L}
= \nabla_{\mathbf{y}} \mathcal{L}
+ \left(\frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)^{\!\top} \nabla_{\mathbf{y}} \mathcal{L}.
\]

The first term is **direct backprop through the skip**; the second is **backprop through** \(\mathcal{F}\).

---

## 4. VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(d\) | Hidden dimension per token |
| \(\mathbf{x}\) | Input to a residual sub-block (e.g. post-norm input or pre-norm depending on wiring) |
| \(\mathcal{F}\) | Residual function (attention block or FFN block, possibly including norm inside) |
| \(\mathbf{y}\) | Output \(\mathbf{x}+\mathcal{F}(\mathbf{x})\) (shape \(d\) or batched \(B\times L \times d\)) |
| \(\mathbf{I}\) | \(d \times d\) identity; **always present** in \(\partial \mathbf{y}/\partial \mathbf{x}\) |
| \(\partial \mathcal{F}/\partial \mathbf{x}\) | Jacobian of the nonlinear stack (attention + projections, etc.) |
| \(L\) | Number of layers (depth); also used for sequence length in attention cost — context disambiguates |
| \(\rho\) | Typical upper bound or spectral-norm scale of \(\|\partial f_\ell/\partial \mathbf{h}\|\) for a plain chain |

---

## 5. Gradient Flow Through One Residual

Vector chain rule for \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})\):

\[
\frac{\partial \mathcal{L}}{\partial x_i}
= \sum_j \frac{\partial \mathcal{L}}{\partial y_j}
\frac{\partial y_j}{\partial x_i}
= \frac{\partial \mathcal{L}}{\partial y_i}
+ \sum_j \frac{\partial \mathcal{L}}{\partial y_j}
\frac{\partial F_j}{\partial x_i}.
\]

So \(\partial \mathbf{y}/\partial \mathbf{x} = \mathbf{I} + \mathbf{J}_\mathcal{F}\) encodes **both** paths. This is the rigorous meaning of “\(\partial \mathcal{L}/\partial \mathbf{x} = \partial \mathcal{L}/\partial \mathbf{y} \times (\mathbf{I} + \partial \mathcal{F}/\partial \mathbf{x})\)” in matrix notation.

**Why this fights vanishing:** Suppose the problematic part of backprop through \(\mathcal{F}\) is weak (e.g. many Jacobians with singular values \(<1\)). The **gradient still receives** \(\nabla_{\mathbf{y}}\mathcal{L}\) scaled by \(\mathbf{I}\) along the skip, so the signal cannot be **multiplied away to zero** by the chain through \(\mathcal{F}\) alone — the skip term remains.

---

## 6. Chained Residuals and Vanishing Gradients

For a **plain** deep network **without** residuals, writing \(\mathbf{h}_{\ell} = f_\ell(\mathbf{h}_{\ell-1})\),

\[
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{0}}
= \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}
\mathbf{J}_L \mathbf{J}_{L-1} \cdots \mathbf{J}_1.
\]

If \(\|\mathbf{J}_\ell\| \le \rho < 1\) in an appropriate norm (e.g. spectral norm), then \(\|\partial \mathcal{L}/\partial \mathbf{h}_0\|\) can scale as \(\mathcal{O}(\rho^L)\): **exponential shrinkage** with depth.

With **residual** blocks \(\mathbf{h}_\ell = \mathbf{h}_{\ell-1} + \mathcal{F}_\ell(\mathbf{h}_{\ell-1})\), the Jacobian of one step is \(\mathbf{I} + \mathbf{J}_{\mathcal{F}_\ell}\). The product structure is more favourable: each step has a persistent identity-like component; deep Transformers are routinely trained with \(L=32\)–\(128\). Empirical spectral analysis often shows gradients remain usable in earlier layers (unlike many plain deep MLPs or early RNNs).

**Caveat:** This does **not** prove \(\|\nabla_{\mathbf{h}_0}\mathcal{L}\|\) is independent of \(L\); pathologies (attention logits, bad init) still exist. Skips **greatly improve** the baseline for depth.

---

## 7. Signal Propagation and Depth Scaling

At initialisation, consider **random weights** and near-linear regime. Heuristic mean-field style analysis asks whether \(\mathrm{Var}(\mathbf{h}_\ell)\) stays \(\Theta(1)\) as \(\ell\) grows.

For \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})\) with \(\mathcal{F}\) scaled to have small output at init (typical with careful init and normalisation),

\[
\|\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle \mathbf{x}, \mathcal{F}(\mathbf{x})\rangle + \|\mathcal{F}(\mathbf{x})\|^2.
\]

If \(\mathcal{F}(\mathbf{x})\) has zero mean correlation with \(\mathbf{x}\) and small norm, \(\mathbb{E}\|\mathbf{y}\|^2 \approx \mathbb{E}\|\mathbf{x}\|^2 + \mathbb{E}\|\mathcal{F}(\mathbf{x})\|^2\). **Depth can add mild variance growth** proportional to \(\sum_\ell \mathbb{E}\|\mathcal{F}_\ell\|^2\), unlike explosive \(\rho^L\) growth in poorly scaled plain nets.

**Role of RMSNorm/LayerNorm:** Pre-norm divides by RMS (or centres/scales in LN), constraining per-token scale before \(\mathcal{F}\), so \(\mathcal{F}\) sees \(\Theta(1)\) inputs at every layer — critical when \(L\gg 1\).

---

## 8. Pre-Norm vs Post-Norm

**Post-norm** (original Transformer, Vaswani et al.): \(\mathbf{y} = \mathrm{Norm}(\mathbf{x} + \mathcal{F}(\mathbf{x}))\).

**Pre-norm** (used in GPT-2+, LLaMA, etc.): \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathrm{Norm}(\mathbf{x}))\).

| Aspect | Post-norm | Pre-norm |
|--------|-----------|----------|
| Where norm sits | After residual sum | Before \(\mathcal{F}\) |
| Signal into \(\mathcal{F}\) | Can be “hotter” / less controlled | Always normalised inputs to \(\mathcal{F}\) |
| Optimisation (deep stacks) | Often harder without warm-up tricks | Usually **more stable** |
| Representation at last layer | Norm closing each block | Norm only inside sublayer; final RMSNorm before LM head |

**Why modern LLMs prefer pre-norm:** Each \(\mathcal{F}_\ell\) always receives **normalised** activations, which keeps Jacobian statistics of attention and MLP more stable across depth. The residual stream \(\mathbf{h}_\ell\) remains a **direct sum** of prior layer contributions (plus norms), matching the “grad highway” story. A final norm (e.g. RMSNorm before `lm_head`) sets scale for logits.

---

## 9. Variance Growth and ResNets

**Connection to ResNets (He et al., 2015):** CNN residual blocks learn \(\mathcal{H}(\mathbf{x})\) by writing \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})\) with \(\mathcal{F}\) representing **residual** mapping. Identity-by-default (**if** \(\mathcal{F}\rightarrow 0\)) eases optimisation when the desired map is near identity.

Transformers inherit the same structural idea in **representation space** (vectors per token), not spatial feature maps:

| ResNet (vision) | Transformer decoder |
|-----------------|---------------------|
| \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})\) | Same |
| \(\mathcal{F}\): conv stack | \(\mathcal{F}\): MHA or FFN (after norm) |
| Deep \(50\)–\(150\) layers | Deep \(32\)–\(128\) blocks |

**Ensemble / path-dropping view:** Residual nets behave like exponential ensembles over subnetworks (Veit et al., 2016); similarly, Transformers can be viewed as many length-\(L\) paths through the stack.

---

### 9.1 INTUITION: Telescope on the Forward Pass (Not the Jacobian Product)

```
If we UNROLL only the SKIP sums (ignoring norm for a moment):

  h_1 = h_0 + F_1(h_0')
  h_2 = h_1 + F_2(h_1') = h_0 + F_1(h_0') + F_2(h_1')
  ...
  h_L = h_0 + Σ_{ℓ=1..L} F_ℓ(h_{ℓ-1}')

Each F_ℓ is applied to a NORMALISED version h_{ℓ-1}' in pre-norm, so this is
not a literal sum of identical maps — but Morally: deeper = sum of L residuals
built on progressively richer h. The INITIAL identity h_0 (embedding output)
still influences h_L through every + in the chain.
```

### 9.2 Backward Through \(L\) Blocks — Structure (Qualitative)

Stack pre-norm layers: \(\mathbf{h}_\ell = \mathbf{h}_{\ell-1} + \mathcal{F}_\ell(\mathrm{Norm}(\mathbf{h}_{\ell-1}))\). Backpropagation computes

\[
\nabla_{\mathbf{h}_{\ell-1}}\mathcal{L}
= \nabla_{\mathbf{h}_\ell}\mathcal{L}
 + (\text{VJP of }\mathcal{F}_\ell\circ\mathrm{Norm})\,\nabla_{\mathbf{h}_\ell}\mathcal{L}.
\]

Unrolling one step gives direct injection of \(\nabla_{\mathbf{h}_L}\mathcal{L}\) into shallower layers through repeated **add** operations from skips — unlike a **pure** composition \(\mathbf{h}_\ell = f_\ell(\mathbf{h}_{\ell-1})\) where only products \(\prod \mathbf{J}_i\) appear.

**Takeaway:** Skips introduce **additive** paths in the VJP; composition introduces **multiplicative** Jacobian products. Depth hurts multiplicative chains (\(\rho^L\)); additive paths preserve a \(\nabla_{\mathbf{h}_L}\mathcal{L}\) contribution analogous to the residual forward **highway**.

### 9.3 When Gradients Can Still Attenuate

Residuals **do not** remove all difficult terms:

- **Softmax** in attention can saturate (small gradients through probabilities).
- **RMSNorm** Jacobian couples coordinates; \(\mathbf{I}\) is on the **residual stream**, not inside normalisation’s re-scaling.
- **Bad learning rates** or numerical overflow still break training.

So the correct claim is: **identity shortcuts prevent the depth-induced multiplicative Jacobian pathology typical of plain deep nets**, not that every backward magnitude stays \(\Theta(1)\).

---

## 10. NUMERICAL EXAMPLES

**Example A — Jacobian of a linear residual:** \(\mathcal{F}(\mathbf{x}) = \mathbf{W}\mathbf{x}\), \(\mathbf{W}\in\mathbb{R}^{d\times d}\).

\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I} + \mathbf{W}.
\]

If \(\mathbf{W} = -\mathbf{I}\) (pathological), \(\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{0}\) — **degenerate** (shows \(\mathcal{F}\) can still cancel the skip in principle). Good init avoids \(\mathbf{W} \approx -\mathbf{I}\).

**Example B — Spectral “plain vs residual” toy:** Suppose each layer’s Jacobian (without skip) has spectral norm \(0.9\). For \(L=32\), \(0.9^{32} \approx 0.034\). A residual step \(\mathbf{I} + \mathbf{J}\) with \(\|\mathbf{J}\|=0.9\) does **not** reduce to \(0.9^{32}\) in the same way — the identity term changes the product structure entirely (full-block Jacobians are \(\mathbf{I}+\mathbf{J}_\ell\), not only \(\mathbf{J}_\ell\)).

**Example C — Pre-norm forward snippet (scalar RMS for one token,d=4):**

Let \(\mathbf{x} = [2, 2, 2, 2]\), RMS\((\mathbf{x}) = \sqrt{4} = 2\), \(\hat{\mathbf{x}} = \mathbf{x}/2 = [1,1,1,1]\). Suppose \(\mathcal{F}\) returns a small residual \([0.1, -0.1, 0, 0.05]\). Then

\[
\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathrm{RMSNorm}(\mathbf{x}))
= [2,2,2,2] + [0.1,-0.1,0,0.05]
= [2.1, 1.9, 2, 2.05].
\]

Gradients mix: \(\partial \mathcal{L}/\partial \mathbf{x}\) includes direct \(\partial \mathcal{L}/\partial \mathbf{y}\) plus corrections from \(\mathcal{F}\circ\)RMSNorm.

**Example D — Inner product growth:** \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})\) with \(\|\mathbf{x}\|=1\), \(\|\mathcal{F}(\mathbf{x})\| = \varepsilon\), and \(\langle \mathbf{x}, \mathcal{F}(\mathbf{x})\rangle = 0\) (orthogonal residual). Then \(\|\mathbf{y}\|^2 = 1 + \varepsilon^2\). For \(\varepsilon = 0.1\): \(\|\mathbf{y}\|\approx 1.005\), only **0.5%** norm inflation. Contrast a chain with \(\|\mathbf{h}_\ell\| = 0.9 \|\mathbf{h}_{\ell-1}\|\): after \(32\) layers, \(\|\mathbf{h}_{32}\| = 0.9^{32} \|\mathbf{h}_0\| \approx 0.034\|\mathbf{h}_0\|\).

**Example E — Chain vs single skip (scalar toy):** Without residual, \(h_{\ell} = 0.95\, h_{\ell-1}\) gives \(h_{32} = 0.95^{32} h_0 \approx 0.198\, h_0\). With \(h_{\ell} = h_{\ell-1} + 0.05\, g(h_{\ell-1})\) and \(|g|\le |h|\), bounds differ: the **additive** term \(h_0\) still appears in \(h_\ell\) when unrolled; multiplicative shrinkage does not apply to that backbone the same way.

---

## 11. Pseudocode

```
# Forward: one pre-norm Transformer sublayer (attention or FFN)

def pre_norm_residual(x, submodule, norm):
    """
    x: [B, L, d]
    submodule: callable; for attention or SwiGLU FFN
    norm: RMSNorm or LayerNorm on last dim
    """
    h = norm(x)
    delta = submodule(h)      # e.g. MHA(h) or FFN(h)
    return x + delta          # residual in representation space


# Backward intuition (what autograd computes)

# Given grad_y = dL/dy of shape [B, L, d]
# y = x + delta
# =>
# dL/dx = grad_y + dL/ddelta * ddelta/dx
# First term is the identity/skip path distributing grad_y to x unchanged.
# Second term flows through submodule and norm.
```

Mathematically, autodiff applies the VJP of \(\mathrm{Norm}\) and \(\mathcal{F}\); the **bias-free copy** \(\mathbf{x}\mapsto \mathbf{y}-\mathcal{F}(\cdot)\) always injects \(\nabla_{\mathbf{y}}\mathcal{L}\) into \(\nabla_{\mathbf{x}}\mathcal{L}\).

---

## 12. COMMON MISTAKES

| ❌ Wrong | ✓ Correct |
|---------|-----------|
| “Residuals **guarantee** \(\|\nabla_{\mathbf{x}}\mathcal{L}\|\) cannot vanish.” | Skips **greatly help**; pathological \(\mathcal{F}\) or loss can still cause issues. The identity **adds** a term, not a lower bound on overall norm. |
| “\(\partial \mathbf{y}/\partial \mathbf{x}\) is always \(\mathbf{I}\).” | Full Jacobian is \(\mathbf{I} + \partial \mathcal{F}/\partial \mathbf{x}\). The **gradient** w.r.t. \(\mathbf{x}\) has two contributions; **partial derivative of y w.r.t. x** includes \(\partial \mathcal{F}/\partial \mathbf{x}\). |
| “Post-norm is identical to pre-norm if you reorganise code.” | Order differs: \(\mathrm{Norm}(\mathbf{x}+\mathcal{F}(\cdot))\) vs \(\mathbf{x}+\mathcal{F}(\mathrm{Norm}(\cdot))\) changes **what** \(\mathcal{F}\) sees and gradient paths. |
| “ResNets and Transformers use different maths.” | Same **residual add** \(\mathbf{y}=\mathbf{x}+\mathcal{F}(\mathbf{x})\); domain-specific \(\mathcal{F}\). |

---

## 13. EXERCISES

1. **Two-layer Jacobian:** For \(\mathbf{h}_1 = f_1(\mathbf{x})\), \(\mathbf{h}_2 = f_2(\mathbf{h}_1)\) without residuals, write \(\partial \mathcal{L}/\partial \mathbf{x}\). Then replace \(\mathbf{h}_1 = \mathbf{x} + f_1(\mathbf{x})\) and expand \(\partial \mathcal{L}/\partial \mathbf{x}\) using \(\partial \mathbf{h}_1/\partial \mathbf{x} = \mathbf{I} + \partial f_1/\partial \mathbf{x}\).

2. **LoRA connection:** Express full fine-tuned weights as \(\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}\). In what sense is \(\mathbf{B}\mathbf{A}\) a **residual** on \(\mathbf{W}_0\)? Relate optimisation difficulty to \(\|\mathbf{B}\mathbf{A}\|\ll \|\mathbf{W}_0\|\).

3. **Norm growth:** Suppose \(\|\mathcal{F}_\ell(\mathbf{h})\| = 0.05 \|\mathbf{h}\|\) and \(\mathbf{h}_{\ell}=\mathbf{h}_{\ell-1}+\mathcal{F}_\ell(\mathrm{Norm}(\mathbf{h}_{\ell-1}))\). Bound \(\|\mathbf{h}_L - \mathbf{h}_0\|\) under simplifying assumptions (e.g. orthogonality between \(\mathbf{h}\) and \(\mathcal{F}\)).

4. **Pre/post:** Draw computational graphs for \(\mathrm{Norm}(\mathbf{x}+\mathcal{F}(\mathbf{x}))\) vs \(\mathbf{x}+\mathcal{F}(\mathrm{Norm}(\mathbf{x}))\). Where does \(\mathbf{x}\) connect directly to the output in each?

5. **Spectral curiosity:** For \(\mathbf{J}=\mathbf{I}+\mathbf{W}\) with symmetric \(\mathbf{W}\), eigenvalues are \(1+\lambda(\mathbf{W})\). How does this contrast with \(\mathbf{J}=\mathbf{W}\) only?

---

**Further reading:** He et al., Deep Residual Learning (2015); Vaswani et al., Attention Is All You Need (2017); Xiong et al., On Layer Normalization in Transformers (2020); Bachlechner et al., ReZero (2021).
