# Sampling Strategies for LLM Generation

Mathematical survey of **stochastic decoding**: **temperature scaling**, **top-\(k\)**, **top-\(p\)** (nucleus), **min-\(p\)**, **repetition penalties**, **typical sampling (information-theoretic)**, and why sampling explores **high-entropy** creative regions that greedy/beam underutilize.

---

## Table of Contents

- [1. Variables](#1-variables)
- [2. Intuition — Shaping the Next-Token Distribution](#2-intuition--shaping-the-next-token-distribution)
- [3. Temperature Scaling](#3-temperature-scaling)
- [4. Top-\(K\) Sampling](#4-top-k-sampling)
- [5. Top-\(P\) (Nucleus) Sampling](#5-top-p-nucleus-sampling)
- [6. Min-\(P\) Filtering](#6-min-p-filtering)
- [7. Repetition Penalty / Logit Processing](#7-repetition-penalty--logit-processing)
  - [7.1 Logit bias decomposition](#71-logit-bias-decomposition)
- [8. Typical Sampling (Brief Information-Theoretic View)](#8-typical-sampling-brief-information-theoretic-view)
  - [8.1 Softmax Jacobian and temperature sensitivity](#81-softmax-jacobian-and-temperature-sensitivity)
  - [8.2 Typical threshold (conceptual recipe)](#82-typical-threshold-conceptual-recipe)
- [9. Why Sampling Beats Greedy/Beam for Diversity](#9-why-sampling-beats-greedybeam-for-diversity)
- [10. Pseudocode](#10-pseudocode)
- [11. Numerical Examples](#11-numerical-examples)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)

---

## 1. Variables

| Symbol | Meaning |
|--------|---------|
| \(z_k\) | Logit for token \(k\) at current step |
| \(T\) | **Temperature** (not sequence length here) |
| \(\tau\) | Sometimes used for temperature; we use \(T\) in formulas **except** where noted |
| \(K\) | Top-\(k\) cutoff count |
| \(p\) | Nucleus mass threshold in \((0,1]\) |
| \(\mathrm{min\_p}\) | Min-\(p\) threshold factor vs running max prob |
| \(H(P)\) | Shannon entropy of distribution \(P\) |
| \(P(w)\) | Final sampling distribution after **all** transforms |

---

## 2. Intuition — Shaping the Next-Token Distribution

```
     Raw logits z
         │
         ▼
   ┌─────────────┐     T<1: sharpen (low entropy)   T>1: flatten (high entropy)
   │  / T soft   │
   └─────────────┘
         │         top-k / top-p / min-p prune unlikely mass
         ▼
   renormalize on survivors ──► sample w
```

**Takeaway:** Sampling is **not** one trick—it is a **pipeline** of monotone / pruning operations followed by **exact sampling** from the resulting distribution.

---

## 3. Temperature Scaling

Let logits \(\{z_k\}_{k\in V}\). Temperature \(T>0\):

\[
P_T(k) = \frac{\exp(z_k / T)}{\sum_{j\in V} \exp(z_j / T)}.
\]

**Limits:**

- \(T\to 0^+\): **one-hot** on \(\arg\max_k z_k\) (greedy).
- \(T\to +\infty\): approaches **uniform** over \(V\).

**Entropy monotonicity (qualitative):** For typical LM logits, increasing \(T\) **raises** next-token entropy \(H(P_T)\) until saturating near uniform.

---

## 4. Top-\(K\) Sampling

Let \(\mathcal{S}_K\) be the set of \(K\) tokens with highest **raw** (or temperature-scaled) probabilities. **Restrict** support to \(\mathcal{S}_K\), then renormalize:

\[
P_{\mathrm{top-}K}(k) \propto \begin{cases}
P_T(k), & k\in \mathcal{S}_K \\
0, & \text{otherwise}
\end{cases}
\]

**Edge case:** Ties and numerics may shuffle boundary tokens—implementation detail.

---

## 5. Top-\(P\) (Nucleus) Sampling

Sort probabilities descending: \(p_{(1)} \ge p_{(2)} \ge \cdots\).

Find smallest prefix index \(m\) such that

\[
\sum_{i=1}^{m} p_{(i)} \ge p_{\mathrm{nucleus}}.
\]

Keep tokens \(\{(1),\ldots,(m)\}\), renormalize.

**Interpretation:** Adaptively widen/shrink candidate set based on **confidence mass**; **flat** distributions keep many tokens, **peaked** distributions auto-restrict.

---

## 6. Min-\(P\) Filtering

Let \(p_{\max} = \max_k P_T(k)\). Keep token \(k\) iff

\[
P_T(k) \ge \mathrm{min\_p} \times p_{\max}.
\]

**Effect:** When the model is **very peaked** (\(p_{\max}\approx 1\)), threshold is strict; when the model is **flat**, absolute floor lowers, preserving many options. This **couples threshold to sharpness**.

---

## 7. Repetition Penalty / Logit Processing

A common **heuristic**:

\[
z'_k =
\begin{cases}
z_k / \rho,& \text{if token }k\text{ appeared recently in context} \\
z_k,& \text{otherwise}
\end{cases}
\]

with \(\rho>1\). Then apply softmax on \(z'\).

**Caveat:** This **breaks** strict probabilistic semantics—it is **not** sampling from the model’s true \(P_\theta\) anymore unless justified as **guided decoding**.

### 7.1 Logit bias decomposition

Any pipeline that maps logits \(z\mapsto z+b(z,\text{context})\) then softmax is equivalent to multiplying **unnormalised** probabilities by **positive** factors \(\exp(b_k)\) that may depend on entire histories. This is a **controlled Markovian logit change** only if \(b_k\) depends on \((k,\text{small finite state})\); repetition modifiers depending on large windows create **non-Markov** decoding **even if the base LM is Markov** of order fixed by context length.

---

## 8. Typical Sampling (Brief Information-Theoretic View)

**Informal idea:** among tokens with sufficiently high \(P(k)\), prefer those whose **information** \(I(k)=-\log P(k)\) lands near **expected information** \(\approx H(P)\)—i.e., **neither** extreme flukes **nor** near-deterministic repeats.

**Motivation:** Extremely improbable tails are often gibberish; extremely high-probability tokens can drive dull loops—**typical set** language borrows from **typicality** in information theory.

### 8.1 Softmax Jacobian and temperature sensitivity

Let \(p = \mathrm{softmax}(z/T)\). For finite differences,

\[
\frac{\partial p_k}{\partial z_j}
= \frac{1}{T} p_k(\delta_{kj} - p_j).
\]

At **low** \(T\), eigenvalues of the Jacobian (**shrink** off-argmax directions) grow in separation—distribution becomes **peaky**. At **high** \(T\), mass equilibrates—**flat**.

### 8.2 Typical threshold (conceptual recipe)

Let \(Z=-\log P_T(k)\) (random information). **Typical sampling** retains token \(k\) if

\[
|Z - H(P_T)| \le \Delta,
\]

or uses **quantile bands** around \(H(P_T)\), then renormalizes. The hyperparameter \(\Delta\) trades tail trimming vs diversity.

---

## 9. Why Sampling Beats Greedy/Beam for Diversity

- **High-entropy regimes** (creative writing, persona) benefit from **stochasticity**: draws from **multi-modal** conditional distributions explore **different basins** of plausible text.
- **Greedy** collapses to a **single mode chain**—often repetitive or locally optimal but globally bland.
- **Beam** enumerates **high-probability** shortlist but still **suppresses** many **valid** medium-probability continuations; it optimizes **probability**, not subjective **novelty**.

**ASCII intuition:**

```
Prob.simplex (triangle sketch):

        high prob mass ●●●   ← greedy/beam hug these peaks
              ○  ○          ← sampling visits varied interior points
```

---

## 10. Pseudocode

```
function temperature(logits z, T):
    return softmax(z / T)

function top_k(logits z, K):
    idx ← indices of K largest z
    mask all other logits → −∞
    return softmax(masked z)

function top_p(probs P, p_mass):
    sort tokens by P descending
    take minimal prefix with cumulative ≥ p_mass
    zero out others; renormalize

function min_p_filter(probs P, min_p):
    pmax ← max P
    keep k where P[k] ≥ min_p * pmax
    zero others; renormalize

function sample_token(P):
    return categorical_draw(P)

# Typical pipeline:
P ← softmax(z / T)
P ← top_p(P, p_nucleus)   # or top_k first; order matters—document your stack!
P ← min_p_filter(P, min_p)
w ← sample_token(P)
```

---

## 11. Numerical Examples

### Example 1 — Temperature effect

Suppose two leading logits: \(z_1=2.0, z_2=1.0\).

- At \(T=1\): \(p_1 \propto e^{2}\approx 7.39,\; p_2 \propto e^{1}\approx 2.72\), \(p_1 \approx 0.73\).
- At \(T=2\): \(p_1 \propto e^{1},\, p_2 \propto e^{0.5}\), closer probabilities (\(p_1 \approx 0.62\)).

### Example 2 — Top-\(p\) nucleus

Sorted probs: \((0.50, 0.25, 0.15, 0.08, 0.02)\).

Cumulative:

- Prefix 1: 0.50 (<0.9)
- Prefix 2: 0.75 (<0.9)
- Prefix 3: **0.90** (meets \(p_{\mathrm{nucleus}}=0.9\))

Keep top **3** tokens.

### Example 3 — Min-\(p\)

If \(\max_k P(k)=0.6\) and \(\mathrm{min\_p}=0.1\), threshold \(=0.06\). Tokens below 0.06 vanish after renormalize.

### Example 4 — Pipeline order matters

Consider \(T=1\), base probs roughly uniform after top-\(p\). Apply **min-\(p\)** *before* nucleus: rare tails may already be removed; **reapplying** top-\(p\) second can yield a **different** support than **nucleus then min-\(p\)**. Always record the **exact commutative diagram** your framework uses.

### Example 5 — Entropy of two-level softmax

For logits \((z,0)\), softmax \(p=(p,1-p)\) with \(p=\frac{e^z}{1+e^z}\). Entropy
\[
H(p)=-p\log p-(1-p)\log(1-p)
\]
is maximized at \(p=\tfrac{1}{2}\) (maximum **uncertainty** on binary choice).

---

## 12. Common Mistakes

- ❌ **Stack top-\(k\) after top-\(p\) without thinking** about double truncation.

  ✓ Document **order**; ablate settings per model family.

- ❌ **Use high temperature to “fix” bad prompting.**

  ✓ Coherence issues often originate in **context**, not decoding alone.

- ❌ **Assume repetition penalty conserves a proper sampling from \(P_\theta\).**

  ✓ It is **biased** decoding—evaluate if that's acceptable.

- ❌ **Set nucleus \(p\) extremely close to 1 on huge vocab.**

  ✓ You may **re-enable** rare junk tokens; pair with min-\(p\) or tail tests.

- ❌ **Confuse entropy of **one** next-token draw with long-sequence diversity.**

  ✓ **Drift** across many steps can still collapse; **block** repetition patterns separately.

---

## 13. Exercises

1. Prove \(\lim_{T\to 0^+} P_T\) concentrates on \(\arg\max z\) assuming unique maximum.

2. Show top-\(K\) can **drop** the true greedy token **never** if you select \(K\) from **post-temperature** logits correctly—when can ties fool this?

3. **Design** a min-\(p\) case that **removes** all tokens except the argmax—when?

4. Express **entropy** \(H(P_T)\) derivative w.r.t. \(T\) qualitatively for 2-token distributions.

5. Compare typical sampling to **entropy clipping**—when might they disagree?

---

### Summary

Sampling reshapes **which modes** you expose from \(\mathbb{R}^{|V|}\) **logit simplices**. Understand your **pipeline order**, measure **perplexity / win-rates / diversity** jointly, and remember: **no free lunch** between creativity and factuality.
