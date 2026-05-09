# Kaplan Scaling Laws (2020)

Power-law scaling of autoregressive language-model loss as a function of model size \(N\), training data size \(D\), and aggregate training compute \(C\). This note develops the Kaplan et al. (2020) empirical laws, contrasts them with Chinchilla (2022), and works concrete numbers.

---

## Table of Contents

- [INTUITION](#intuition)
- [VARIABLES & NOTATION](#variables--notation)
- [LAW 1: Loss vs Parameters \(L(N)\)](#law-1-loss-vs-parameters-ln)
- [LAW 2: Loss vs Data \(L(D)\)](#law-2-loss-vs-data-ld)
- [LAW 3: Loss vs Compute \(L(C)\)](#law-3-loss-vs-compute-lc)
- [Irreducible Loss \(E\)](#irreducible-loss-e)
- [Log–Log Geometry and Exponent Algebra](#loglog-geometry-and-exponent-algebra)
- [Joint Scaling and Resource Allocation](#joint-scaling-and-resource-allocation)
- [Kaplan vs Chinchilla](#kaplan-vs-chinchilla)
- [NUMERICAL EXAMPLES](#numerical-examples)
- [Pseudocode](#pseudocode)
- [Fitting, Breaks, and Extrapolation](#fitting-breaks-and-extrapolation)
- [COMMON MISTAKES](#common-mistakes)
- [EXERCISES](#exercises)
- [REFERENCES](#references)

---

## INTUITION

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  KAPLAN VIEW (2020):  Loss is a SMOOTH POWER LAW in resources            │
    │                                                                          │
    │      L ──┐                                    Chinchilla later showed:   │
    │          │\                              optimal D/N is MUCH larger than  │
    │          │ \___                           Kaplan implied for fixed C.    │
    │          │     \____                                                      │
    │          └──────────\──────────►  log N or log D or log C               │
    │                                                                          │
    │  In LOG–LOG space, straight lines  ⟹  multiplicative gains feel "dull": │
    │       2× resource  ⟹  only ~few % loss drop (small exponents α).         │
    └─────────────────────────────────────────────────────────────────────────┘

         N, D, C  ──power laws──►  predictable loss  ──►  budgets & roadmaps

    Kaplan: fit  L - E  ∝  (scale / resource)^α   on massive sweeps; extrapolate.
```

**One-sentence summary:** After subtracting a data- and task-dependent irreducible baseline \(E\), the reducible cross-entropy behaves like \(L - E \propto N^{-\alpha_N}\), \(D^{-\alpha_D}\), \(C^{-\alpha_C}\) in appropriate regimes—so equal **percentage** steps in \(\log N\), \(\log D\), or \(\log C\) produce **predictable** relative loss improvements.

---

## VARIABLES & NOTATION

| Symbol | Meaning | Typical Kaplan magnitude (order) |
|--------|---------|-----------------------------------|
| \(L\) | Final (or best) cross-entropy **nats/token** after training under a protocol | varies by setup |
| \(N\) | Non-embedding parameter count | \(10^7\)–\(10^{12+}\) |
| \(D\) | Unique (or effective) training **tokens** seen | \(10^9\)–\(10^{13+}\) |
| \(C\) | **6ND** FLOPs proxy: forward+backward through dense layers | matches Kaplan compute axis |
| \(E\) | **Irreducible** loss: Bayes / best possible under data + task | > 0; shifts the power-law line |
| \(N_c, D_c, C_c\) | Scale constants from fitting \(\log(L-E)\) vs \(\log(\cdot)\) | paper-specific |
| \(\alpha_N\) | Slope in \(\log(L-E)\) vs \(\log N\) (fit region) | ≈ **0.076** |
| \(\alpha_D\) | Slope in \(\log(L-E)\) vs \(\log D\) | ≈ **0.095** |
| \(\alpha_C\) | Slope in \(\log(L-E)\) vs \(\log C\) | ≈ **0.050** |

**FLOPs model used throughout Kaplan-style scaling:** for a dense Transformer, pretraining compute is often summarized as

\[
C \approx 6 N D
\]

(six per-parameter, per-token multiply-adds for forward+backward, ignoring attention in the leading term). This links **joint** choices \((N,D)\) to a single compute budget.

---

## LAW 1: Loss vs Parameters \(L(N)\)

Fix data and training procedure; sweep model size \(N\). Empirically (Kaplan),

\[
L(N) \;=\; E \;+\; A_N \left(\frac{N_c}{N}\right)^{\alpha_N}
\]

or equivalently, focusing on the reducible part \(\tilde{L} = L - E\),

\[
\tilde{L}(N) \;\propto\; N^{-\alpha_N}.
\]

**Local sensitivity (elasticity):** differentiating \(\log \tilde{L}\) w.r.t. \(\log N\),

\[
\frac{\mathrm{d} \log \tilde{L}}{\mathrm{d} \log N} = -\alpha_N
\quad\Rightarrow\quad
\frac{\Delta \tilde{L}}{\tilde{L}} \approx -\alpha_N \,\frac{\Delta N}{N}.
\]

So a **1% increase in \(N\)** yields roughly a **\(0.01 \alpha_N\)** relative drop in reducible loss. With \(\alpha_N \approx 0.076\), **doubling** \(N\) gives

\[
\frac{\tilde{L}(2N)}{\tilde{L}(N)} \approx 2^{-\alpha_N} \approx 2^{-0.076} \approx 0.949,
\]

about **5.1%** relative improvement in \(\tilde{L}\)—not halving, not a giant jump.

---

## LAW 2: Loss vs Data \(L(D)\)

Fix model and procedure; vary training tokens \(D\). Write

\[
L(D) \;=\; E \;+\; A_D \left(\frac{D_c}{D}\right)^{\alpha_D}.
\]

Here \(\alpha_D \approx 0.095\) is **larger** than \(\alpha_N\) in Kaplan fits: **per equal multiplicative increase**, data scaling bought slightly more reducible loss reduction than width/depth scaling **in those fits**—an important hint that neither axis should be starved naively.

**Doubling data:** \(\tilde{L}(2D)/\tilde{L}(D) \approx 2^{-\alpha_D} \approx 2^{-0.095} \approx 0.937\), about **6.3%** relative improvement in \(\tilde{L}\).

---

## LAW 3: Loss vs Compute \(L(C)\)

When both \(N\) and \(D\) move together along training runs matched on compute, Kaplan also reports a law of the form

\[
L(C) \;=\; E \;+\; A_C \left(\frac{C_c}{C}\right)^{\alpha_C},
\quad \alpha_C \approx 0.050.
\]

**Doubling compute** along the empirical frontier (as **measured** in their study):

\[
\frac{\tilde{L}(2C)}{\tilde{L}(C)} \approx 2^{-\alpha_C} \approx 2^{-0.05} \approx 0.966,
\]

about **3.4%** relative improvement. This is **smaller** per doubling than raw \(N\) or raw \(D\) sweeps because the joint frontier is a **coupled** problem: not all of the extra compute is deployed in the "best" marginal direction unless you optimize the \((N,D)\) split.

---

## Irreducible Loss \(E\)

```
    L
    │
    │  ┌────────  empirical curve
    │ /
    │/________________________ E  ←  floor from noise, Bayes error, task mismatch
    └──────────────────────────────►  N, D, or C
```

**Definition (operational):** \(E\) is the intercept approached as resources \(\to\infty\) under a **fixed data distribution and tokenizer**. It is **not** zero for natural language: ambiguity, rare facts, and imperfect tokenization all contribute.

**Algebra with \(E\):** fits are done in log-space on \(\tilde{L} = L - E\). Mis-estimating \(E\) tilts slopes and breaks extrapolation—especially when \(L\) is close to \(E\).

---

## Log–Log Geometry and Exponent Algebra

For \(\tilde{L} = A X^{-\alpha}\), taking logs (any base \(b\)):

\[
\log_b \tilde{L} = \log_b A - \alpha \log_b X.
\]

A plot of \(\log \tilde{L}\) vs \(\log X\) is a **line** with slope \(-\alpha\). Two models with sizes \(N_1, N_2\):

\[
\frac{\tilde{L}(N_1)}{\tilde{L}(N_2)} = \left(\frac{N_2}{N_1}\right)^{\alpha_N}.
\]

**Example:** Ratio for \(N_1 = 7\times 10^9\), \(N_2 = 70\times 10^9\):

\[
\frac{\tilde{L}(70\text{B})}{\tilde{L}(7\text{B})} \approx \left(\frac{1}{10}\right)^{\alpha_N} = 10^{-0.076} \approx 0.84.
\]

Reducible loss at 70B is ~**84%** of that at 7B (same data protocol) if the pure power law holds—**16%** relative reduction.

---

## Joint Scaling and Resource Allocation

Kaplan combines constraints to reason about **fixed compute** \(C \approx 6ND\). A useful **back-of-envelope** ansatz used in discussions (not unique) writes effective reducible loss as a sum of "model-difficulty" and "data-difficulty" pieces raised to compatible powers. A schematic joint form:

\[
\tilde{L}(N,D)^{1/\alpha_D} \;\approx\; \underbrace{\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}}_{\text{underparameterization}} \;+\; \underbrace{\frac{D_c}{D}}_{\text{data-limited}}.
\]

**Balancing inner terms** at stationarity suggests a scaling relation between optimal \(N^\star\) and \(D^\star\):

\[
\left(\frac{N_c}{N^\star}\right)^{\alpha_N/\alpha_D} \sim \frac{D_c}{D^\star}
\quad\Rightarrow\quad
N^\star \;\propto\; D^{\star\,\alpha_D/\alpha_N}.
\]

With \(\alpha_D/\alpha_N \approx 0.095/0.076 \approx 1.25\), **larger data** wants **superlinearly** larger model in this heuristic—yet Kaplan's **reported compute-optimal frontier** in the paper favored **very large models** paired with less data than later work (see next section). The tension is precisely where **Chinchilla** intervened: their sweeps and stopping criteria implied **much larger \(D/N\)** ratios at compute optimality.

**Interpretation for planners:** power laws are excellent **qualitative** guides—marginal returns shrink smoothly—but **joint optima** depend on training **length**, **architecture**, **noise**, and what is held fixed across runs.

---

## Kaplan vs Chinchilla

| Aspect | Kaplan (2020) empirical frontier | Chinchilla (2022) revision |
|--------|----------------------------------|----------------------------|
| Central claim | Smooth scaling laws + early optimal-allocations guidance | **IsoFLOP** curves; \(D \approx 20 N\) tokens at optimum for many setups |
| Fixed-\(C\) split | Historically leaned **param-heavy**, **data-light** vs Chinchilla | Roughly **\(N \propto C^{0.5}\), \(D \propto C^{0.5}\)** (exponents ~0.49–0.51 in their fits) |
| Why differ? | Training durations, data loading, incomplete sweeps of \(D\) at each \(N\) | Joint grid over \((N,D)\) with matched **final** loss at **fixed** compute |

**Takeaway for this chapter:** Kaplan's **scalar** laws \(L(N)\), \(L(D)\), \(L(C)\) remain the vocabulary of scaling; Chinchilla updates the **coupled** optimum **under revised protocols**. Do not conflate **per-axis** exponents with **joint** compute optimality.

---

## NUMERICAL EXAMPLES

Assume a toy reducible baseline \(\tilde{L}(N_{\text{ref}})\) at \(N_{\text{ref}}=1.3\times 10^{10}\) (13B) and \(\alpha_N=0.076\) for ratios (ignoring \(E\) for simplicity).

1. **\(N = 1.75\times 10^{11}\) (175B)** vs 13B:

\[
\frac{\tilde{L}(175\text{B})}{\tilde{L}(13\text{B})} \approx \left(\frac{13}{175}\right)^{0.076} \approx (0.0743)^{0.076} \approx 0.80.
\]

~**20%** reducible loss reduction from a **13B → 175B** jump under a pure law (actual runs deviate near floors).

2. **Compute check with \(C \approx 6ND\):** if \(N=6.7\times 10^{10}\) and \(D=3.0\times 10^{11}\) tokens,

\[
C \approx 6 \times 6.7\times 10^{10} \times 3.0\times 10^{11} \approx 1.2\times 10^{23}\ \text{FLOPs}.
\]

Varying only one of \(N\) or \(D\) while holding the other fixed moves you **off** the iso-loss manifold in ways the single-variable laws summarize separately.

3. **Log-log slope verification:** A decade in \(N\) changes \(\tilde{L}\) by \(10^{-0.076} \approx 0.84\times\)—about **16%** reducible loss drop per **10×** params (not per +10 params).

---

## Pseudocode

```
# Fit reducible loss power law on log-log grid (base b > 1)
function fit_power_law(N[], L[], E):
    # L: measured losses; E: assumed irreducible floor (estimated jointly if needed)
    for i in 0..len(N)-1:
        y[i] = log_b(L[i] - E)
        x[i] = log_b(N[i])
    (intercept, slope) = LINEAR_REGRESSION(x, y)
    alpha_N = -slope
    N_c = b^(intercept / alpha_N)
    return (alpha_N, N_c)


# Compare two model sizes under the same data regime
function relative_reducible_loss(N1, N2, alpha_N):
    return (N2 / N1) ** alpha_N   # equals L_tilde(N1)/L_tilde(N2) if only N changes


# Joint FLOPs accounting (dense-model proxy)
function flops_forward_backward(N, D):
    return 6 * N * D
```

---

## Fitting, Breaks, and Extrapolation

```
  log(L - E)
      │  ┌──────────────  Region A: power law (healthy)
      │ /
      │/_________________  bend: breaking news / new bottleneck
      │                    (routing, data quality, tokenizer)
      └────────────────────────► log N
```

**Joint estimation of \((E,\alpha,A)\):** minimizing \(\sum_i \big(\log(L_i - E) - \log A + \alpha \log N_i\big)^2\) is **nonlinear** in \(E\). Practitioners either sweep \(E\) on a grid or optimize with positivity constraint \(L_i > E\). Wrong \(E\) flattens slopes.

**Breaks:** power laws are empirical. New data ingredients (tool use, retrieval, synthetic), mixture-of-experts (MoE) routing noise, or changed **token/byte ratio** can change the effective \(\alpha\)’s.

**Risk:** extrapolating from \(10^{9}\)–\(10^{11}\) params to \(10^{13}\) is a bet that the **same** data regime and optimization landscape persist.

---

## COMMON MISTAKES

- ❌ Treat \(L(N)=(N_c/N)^{\alpha_N}\) as exact at small \(N\) or when \(L \approx E\).  
  ✓ Fit **\(\tilde{L}=L-E\)** and validate **log-log linearity** only where data support exists.

- ❌ Use single-axis \(\alpha\)’s to pick \((N,D)\) at fixed \(C\) without joint experiments.  
  ✓ Use **isoFLOP** slices (Chinchilla style) for allocation; keep Kaplan scalars as **marginal** intuition.

- ❌ Forget \(C \approx 6ND\) is a **proxy** (attention, sparse MoE, hardware change scaling).  
  ✓ Re-scale \(C\) or introduce correction terms when architecture deviates.

- ❌ Confuse **tokens seen** \(D\) with **unique corpus size** or **epochs**.  
  ✓ Be explicit whether \(D\) counts **unique** tokens, **total passes**, or **effective** tokens with dropout.

- ❌ Assume slopes are **universal constants** across domains (code vs web vs math).  
  ✓ Re-fit; exponents and \(E\) shift with distribution and tokenizer.

---

## EXERCISES

1. **Algebra:** Show \(\tilde{L}(kN)/\tilde{L}(N) = k^{-\alpha_N}\). For \(k=10\) and \(\alpha_N=0.076\), compute the numeric ratio.

2. **Joint reasoning:** Fix \(C=6\times 10^{23}\). If you naively set \(D=C/(6N)\), plot how \(D\) changes as \(N\) sweeps \(10^9\)–\(10^{12}\). Where does \(D\) dip below **1 epoch** of a 300B-token corpus—what does that imply about **data repetition**?

3. **Chinchilla contrast:** Explain in words why **undertraining large \(N\)** could make empirical \(L(N)\) curves look **better** for big models than a fully converged isoFLOP study would—link to **early stopping** bias.

4. **Floor effects:** If \(E=1.5\) nats/token and measured \(L=1.7\), what fraction of loss is **reducible**? How does a 10% drop in \(\tilde{L}\) translate to a drop in **total** \(L\)?

5. **Design:** Suppose \(\alpha_D > \alpha_N\). Does that automatically imply you should always prefer data over width at fixed \(C\)? Defend using **joint** scaling rather than one-line reasoning.

---

## REFERENCES

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020).  
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022).
