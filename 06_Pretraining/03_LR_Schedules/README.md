# Learning Rate Schedules for LLM Pretraining

*Warmup, cosine decay, WSD, batch scaling, and the LLaMA-3 trajectory*

---

## Table of Contents

1. [Overview](#1-overview)
2. [INTUITION: Loss Landscape and Step Size](#2-intuition-loss-landscape-and-step-size)
3. [VARIABLES](#3-variables)
4. [Why Linear Warmup for Adam](#4-why-linear-warmup-for-adam)
5. [Cosine Decay — Full Formula and Properties](#5-cosine-decay--full-formula-and-properties)
6. [WSD: Warmup-Stable-Decay](#6-wsd-warmup-stable-decay)
7. [Learning Rate and Batch Size (Linear Scaling)](#7-learning-rate-and-batch-size-linear-scaling)
8. [LLaMA-3 Schedule (Typical Pattern)](#8-llama-3-schedule-typical-pattern)
8b. [Adam Effective Step and Warmup (Sketch)](#8b-adam-effective-step-and-warmup-sketch)
9. [Too High vs Too Low Learning Rate](#9-too-high-vs-too-low-learning-rate)
10. [NUMERICAL EXAMPLES](#10-numerical-examples)
11. [Pseudocode](#11-pseudocode)
12. [COMMON MISTAKES](#12-common-mistakes)
13. [EXERCISES](#13-exercises)
14. [Supplement: Loss Curvature](#14-supplement-loss-curvature)

---

## 1. Overview

The **learning rate** \(\eta\) controls the magnitude of parameter updates. In large-scale **LLM pretraining**, \(\eta\) is almost never fixed for the entire run: we **warm up** from a small \(\eta\) to \(\eta_{\max}\), optionally hold **stable**, then **decay** toward \(\eta_{\min}\).

This note gives the **cosine** schedule in the form used in the literature, explains **WSD**, links **batch size** to \(\eta\) via **linear scaling**, sketches **LLaMA-3**-style hyperparameters (illustrative), and analyses **divergence** and **under-utilisation**.

---

## 2. INTUITION: Loss Landscape and Step Size

```
┌──────────────────────────────────────────────────────────────────┐
│  SGD-like update:   θ_{t+1} = θ_t − η · g_t                        │
│                                                                    │
│  g_t = noisy gradient of minibatch loss                              │
│                                                                    │
│  η too LARGE  →  overshoot valleys, oscillate, even EXPLODE loss   │
│                    Loss                                              │
│                     │    ╱╲   ╱╲                                     │
│                     │   ╱  ╲ ╱  ← unstable                         │
│                     └──╱──────── steps                               │
│                                                                    │
│  η too SMALL  →  progress ∝ η  →  waste compute / never finish       │
│                    Loss                                              │
│                     │╲                                             │
│                     │ ╲_______ slow crawl                            │
│                     └──────────── steps                              │
│                                                                    │
│  GOOD SCHEDULE: start gentle (warmup), learn fast (stable),         │
│                 end gentle (decay) to settle in flat minima         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(t\) | Global optimizer step (or token-normalised step — follow codebase) |
| \(T\) | Total number of scheduled steps **after** warmup, or full run — **define per implementation** |
| \(T_w\) | Warmup duration (steps until \(\eta_{\max}\) reached in linear warmup) |
| \(T_s\) | (WSD) start of decay phase — end of **stable** phase |
| \(\eta_{\max}\) | Peak learning rate |
| \(\eta_{\min}\) | Floor at cycle end |
| \(\eta(t)\) | Learning rate at step \(t\) |
| \(B\) | Total batch size (tokens or sequences — be consistent) |
| \(B_{\mathrm{ref}}\) | Reference batch used to pick \(\eta_{\max}\) |

---

## 4. Why Linear Warmup for Adam

Adam maintains **moments** \(\mathbf{m}_t, \mathbf{v}_t\) (biased estimates of gradient and its square). At \(t=0\), bias-correction denominators are small → early steps can produce **very large effective updates** if \(\eta\) starts at full \(\eta_{\max}\).

**Linear warmup:** for \(t \le T_w\),

\[
\eta(t) = \eta_{\max} \cdot \frac{t}{T_w}.
\]

Properties:

- \(\eta(0)=0\): no wild first step from random init.
- Gradually exposes the optimiser to full-sized updates once moments stabilise.
- Standard in GPT-scale and LLaMA-scale recipes (often \(T_w\) in low thousands of **steps** — check each model card).

**Interaction:** Warmup is especially important when using **large** \(\eta_{\max}\), **Adam** (or AdamW), and **large batch** after scaling.

---

## 5. Cosine Decay — Full Formula and Properties

After warmup, a common choice is **cosine annealing** to \(\eta_{\min}\). One widely used form (single cycle, \(t'\) measuring progress in the decay segment):

Let \(t\) be the current step index, \(T_w\) warmup length, \(T\) **total training steps**. Define decay progress \(\tau \in [0,1]\):

\[
\tau = \frac{t - T_w}{T - T_w}, \quad \tau \in [0,1].
\]

Then

\[
\eta(t) = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\bigl(1 + \cos(\pi \tau)\bigr)
\quad \text{for } t \ge T_w.
\]

**Checks:**

- \(\tau=0\) (\(t=T_w\)): \(\cos(0)=1 \Rightarrow \eta = \eta_{\min} + (\eta_{\max}-\eta_{\min}) = \eta_{\max}\).
- \(\tau=1\) (\(t=T\)): \(\cos(\pi)=-1 \Rightarrow \eta = \eta_{\min}\).

Some codebases fold warmup into a cosine on \([0,T]\) directly; **always align** \(\eta_{\max}\) location with your **implementation**.

**Smoothness:** Cosine has **zero derivative** at \(\tau=1\) (since \(\sin(\pi)=0\)), unlike piecewise-constant decay — gentler end-phase than staircase schedules.

---

## 6. WSD: Warmup-Stable-Decay

```
LR
 η_max ───┬───────── Warmup ─────────┬──── Stable ────┬── Decay ──┐
          │         /                 │                │          │
          │        /                  │  flat          │  cos or  │
 η_min ───┴───────┴───────────────────┴────────────────┴── linear ─┘
          0       T_w                 T_s              T_end
```

**Phases:**

1. **Warmup** \([0,T_w]\): \(\eta\) ramps from \(0\) or \(\eta_{\mathrm{start}}\) to \(\eta_{\max}\) (often linear).
2. **Stable** \([T_w, T_s]\): \(\eta(t) = \eta_{\max}\) — **constant** high learning rate for bulk of tokens.
3. **Decay** \([T_s, T_{\mathrm{end}}]\): cosine or linear to \(\eta_{\min}\).

**Why WSD matters:** If you **extend** training (more tokens than originally budgeted), you can **lengthen the stable phase** without recomputing the entire schedule from scratch — flexible for large industrial runs.

---

## 7. Learning Rate and Batch Size (Linear Scaling)

**Heuristic (Goyal et al., 2017; many follow):** When increasing **total minibatch size** from \(B_{\mathrm{ref}}\) to \(B\) **without changing model or data**, try

\[
\eta(B) \approx \eta_{\mathrm{ref}} \cdot \frac{B}{B_{\mathrm{ref}}},
\]

**linear scaling rule**, often with a **warmup** after batch changes.

**Intuition:** Larger batches average gradient noise \(\propto 1/\sqrt{B}\) for SGD; to keep **signal per step** comparable, step size may grow with \(B\). **Reality:** Adam dynamics differ; rule is **not exact** — always validate loss curves.

**Square-root scaling** \(\eta \propto \sqrt{B/B_{\mathrm{ref}}}\) is another common variant (less aggressive).

**Critical constraint:** Memory limits **per-device** batch; global \(B\) uses **data-parallel** replication. Schedule \(\eta\) relative to **global** \(B\).

---

## 8. LLaMA-3 Schedule (Typical Pattern)

Published LLaMA-3 training uses **large token counts** and **distributed** optimisation. **Illustrative** numbers from public reports and community replications (verify against official model card for your clone):

| Hyperparameter | Order of magnitude |
|----------------|-------------------|
| Peak LR \(\eta_{\max}\) | \(\sim 3\times 10^{-4}\) for some 8B-scale runs |
| Warmup steps | \(\sim 10^3\)–\(10^4\) steps (not universal) |
| Cosine decay or WSD-style tail | Yes |
| Weight decay (AdamW) | paired with LR separately — see Adam chapter |

Exact \(\eta(t)\) tables are **release-specific**; when reproducing, **start** from published configs and tune loss spikes.

---

### 8b. Adam Effective Step and Warmup (Sketch)

Adam’s bias-corrected update scales like \(\eta \cdot \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t}+\epsilon)\). Early in training, \(\hat{\mathbf{v}}_t\) can be **small** (few samples) while \(\hat{\mathbf{m}}_t\) is **noisy** → **effective step** can be larger than the naive \(\eta \|\mathbf{g}\|\) suggests. Warmup **limits** \(\eta\) until \(\mathbf{v}_t\) stabilises.

**Order-of-magnitude view:** If the **signal-to-noise** ratio of \(\mathbf{g}_t\) improves after a few thousand steps, raising \(\eta\) **too early** injects disproportionately noisy directions into \(\theta\).

```
┌──────────────────────────────────────────────────────────────────┐
│  EFFECTIVE UPDATE NORM (cartoon)                                  │
│                                                                  │
│   ‖Δθ‖   │     warmup caps spike at t small                       │
│          │  ╭──╮                                                 │
│          │ ╱    ╲_______  stable η at η_max                       │
│          │╱                                                  ╲  decay
│          └────────────────────────────────────────────────────────► t
│             ↑ bias correction + moment build-up region           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Too High vs Too Low Learning Rate

**Too high \(\eta_{\max}\):**

- Loss spikes or **NaN**s (especially with FP16 without loss scaling, or borderline BF16 ranges in poorly scaled models).
- **Optimizer divergence:** exponential growth of activation norms in bad regimes.
- **Instability** early if warmup insufficient — moments not ready for large steps.

**Too low \(\eta\) (or \(\eta_{\min}\) stuck high in wrong schedule):**

- **Underfitting** — same compute, worse loss; training **plateaus** early.
- **Wasted FLOPs** — you pay for full forward/backward but make **tiny** progress per step.

**Decay phase:** Near \(\eta_{\min}\), small updates help **settle** in flatter regions; stopping at high \(\eta\) can leave loss **noisy**.

---

## 10. NUMERICAL EXAMPLES

**Example A — Warmup only:** \(T_w=1000\), \(\eta_{\max}=3\times 10^{-4}\).

\[
\eta(500) = 3\times 10^{-4} \cdot \frac{500}{1000} = 1.5\times 10^{-4}.
\]

**Example B — Cosine (no separate warmup in formula for clarity):** Let \(T_w=0\), \(T=10^5\), \(t=5\times 10^4\), \(\eta_{\min}=3\times 10^{-5}\), \(\eta_{\max}=3\times 10^{-4}\).

\[
\tau = \frac{5\cdot 10^4}{10^5} = 0.5,\quad \cos(\pi/2)=0 \Rightarrow \eta = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min}) = \frac{\eta_{\max}+\eta_{\min}}{2}.
\]

Numerically: \((3\times 10^{-4} + 3\times 10^{-5})/2 = 1.65\times 10^{-4}\).

**Example C — Linear scaling:** Reference \(B_{\mathrm{ref}}=4\text{M}\) tokens/step, \(\eta_{\mathrm{ref}}=3\times 10^{-4}\). New \(B=8\text{M}\).

\[
\eta_{\mathrm{new}} \approx 3\times 10^{-4} \cdot \frac{8}{4} = 6\times 10^{-4}.
\]

**Reality check:** Double LR can destabilise — often use **gradual** batch increase or **sqrt scaling** instead.

**Example D — WSD timeline:** \(T_w=2000\), \(T_s=50{,}000\), \(T_{\mathrm{end}}=100{,}000\).

- \(t=1000\): still **warmup** → \(\eta = \eta_{\max}/2\) (linear to max at 2000).
- \(t=10000\): **stable** → \(\eta=\eta_{\max}\).
- \(t=75000\): **decay** → \(\tau=(75000-50000)/(50000)=0.5\) in cosine segment → \(\eta=(\eta_{\max}+\eta_{\min})/2\) if cosine.

**Example E — Minimum usable \(\eta\):** Suppose loss improvement per step \(\Delta \mathcal{L} \propto \eta\) in a noisy quadratic bowl. Halving \(\eta\) **doubles** steps to reach the same \(\mathcal{L}\) — **linear waste** in \(\eta\) if nothing else limits convergence.

---

## 11. Pseudocode

```python
def lr_cosine_step(t, T_w, T_end, eta_min, eta_max):
    """t: 1..T_end. Linear warmup then cosine to eta_min."""
    if t < T_w:
        return eta_max * (t + 1) / T_w   # or t/T_w depending on 0/1 indexing
    tau = (t - T_w) / max(1, T_end - T_w)
    tau = min(max(tau, 0.0), 1.0)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * tau))


def lr_wsd(t, T_w, T_s, T_end, eta_min, eta_max, decay="cosine"):
    if t < T_w:
        return eta_max * (t + 1) / T_w
    if t < T_s:
        return eta_max
    # decay phase
    tau = (t - T_s) / max(1, T_end - T_s)
    tau = min(max(tau, 0.0), 1.0)
    if decay == "cosine":
        return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * tau))
    return eta_max * (1.0 - tau) + eta_min * tau   # linear fall-back
```

---

## 12. COMMON MISTAKES

| ❌ Wrong | ✓ Correct |
|---------|-----------|
| “Cosine formula is unique.” | Many variants: cosine restarts, cyclical, different \(\tau\) definitions. Match code. |
| “Linear scaling always works.” | Empirical; often needs **longer warmup** or **sqrt** scaling; watch loss. |
| “\(\eta_{\min}=0\) is best.” | Very small \(\eta\) near end can help but **zero** may freeze useful motion; use small positive floor. |
| “Schedule in tokens = schedule in steps.” | Convert with **global batch**: steps = tokens / batch_size. |

---

## 13. EXERCISES

1. With \(T_w=2000\), \(T_{\mathrm{end}}=100000\), \(\eta_{\max}=3\times 10^{-4}\), \(\eta_{\min}=3\times 10^{-5}\), compute \(\eta(t)\) at \(t=1000\), \(t=50000\), \(t=100000\) using the cosine-after-warmup recipe.

2. Explain **why** cosine endpoints have **flat derivative** at full cycle end and how that differs from **hard step decay**.

3. You extend pretraining by **50%** more tokens. Compare adjusting a **fixed-T cosine** vs **extending WSD stable phase**.

4. Derive the **halfway** cosine value (algebraically) for general \(\eta_{\min},\eta_{\max}\).

5. **Batch doubling** with instability: propose a mitigation hierarchy (warmup lengthen, \(\eta\) scaling factor \(<2\), gradient clipping).

6. **Cosine derivative:** Show \(\mathrm{d}\eta/\mathrm{d}\tau = -\tfrac{\pi}{2}(\eta_{\max}-\eta_{\min})\sin(\pi\tau)\). Why is \(\tau=0\) and \(\tau=1\) special?

7. **Token vs step scheduling:** For **1B tokens**, global batch **4M tokens/step**, how many **steps**? If cosine uses **step** axis, convert **token budgets** accordingly.

---

## 14. Supplement: Loss Curvature

Near a **smooth** minimum, loss \(\mathcal{L}(\theta)\approx \mathcal{L}_\star + \tfrac{1}{2}(\theta-\theta_\star)^{\mathsf T}\mathbf{H}(\theta-\theta_\star)\) with Hessian \(\mathbf{H}\). **Ideal** \(\eta \sim 2/\lambda_{\max}(\mathbf{H})\) for GD (upper bound culture). **Adam** adapts per-parameter scales; \(\eta_{\max}\) is still **global knob** — if \(\eta_{\max}\) is far beyond the **stable** range, iterates may **diverge** despite adaptation.

In **LLM pretraining**, \(\mathbf{H}\) is **not** observed explicitly; practitioners tune \(\eta_{\max}, T_w\) on **small** proxies (scaled-down models or short runs) before committing exaFLOP budgets.

---

**References:** Loshchilov & Hutter, SGDR (2016); Goyal et al., Accurate Large Minibatch SGD (2017); Smith, WSD blog/paper; LLaMA / LLaMA 3 technical reports.
