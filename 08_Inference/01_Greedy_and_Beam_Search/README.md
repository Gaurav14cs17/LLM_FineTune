# Greedy Decoding and Beam Search

Formal analysis of **left-to-right** autoregressive decoding: **greedy** argmax stacking, **beam search** over partial hypotheses, **length normalization**, complexity, connection to **MAP / approximate inference**, and when greedy is adequate versus harmful.

---

## Table of Contents

- [1. Variables](#1-variables)
- [2. Intuition — Local vs Global Optimality](#2-intuition--local-vs-global-optimality)
- [3. Greedy Decoding](#3-greedy-decoding)
- [4. Beam Search](#4-beam-search)
- [5. Length Normalization](#5-length-normalization)
- [6. Why Greedy Is Suboptimal](#6-why-greedy-is-suboptimal)
- [7. Complexity Analysis](#7-complexity-analysis)
- [8. MAP View and Search](#8-map-view-and-search)
- [9. When Greedy Suffices vs When Beam Helps](#9-when-greedy-suffices-vs-when-beam-helps)
- [9A. Log-sum structure and why “local argmax” is not global MAP](#9a-log-sum-structure-and-why-local-argmax-is-not-global-map)
- [10. Pseudocode](#10-pseudocode)
- [10A. Worked micro-beam (illustrative)](#10a-worked-micro-beam-illustrative)
- [11. Numerical Examples](#11-numerical-examples)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)

---

## 1. Variables

| Symbol | Meaning |
|--------|---------|
| \(V\) | Vocabulary |
| \(|V|\) | Vocabulary size |
| \(y_{1:n}\) | Output token sequence of length \(n\) |
| \(P_\theta(y_t \mid y_{<t}, x)\) | Conditional next-token distribution |
| \(s(y_{1:t})\) | **Cumulative score** of prefix (often sum of \(\log P\)) |
| \(B\) | Beam width |
| \(\alpha\) | Length normalization exponent in beam scoring |
| \(n\) | Generated length (EOS handling omitted for brevity) |

---

## 2. Intuition — Local vs Global Optimality

```
GREEDY (myopic):
  step t: pick w_t = argmax P(w | prefix)
          ▲
          └── maximises EACH step — NOT necessarily the PRODUCT along the path

BEAM (keeps contenders):
  maintain B best PARTIAL sequences at each step
          ▼
  may recover a path that looked worse at t=1 but wins overall
```

---

## 3. Greedy Decoding

At each time \(t\),

\[
w_t^\star = \arg\max_{w\in V} P_\theta(w \mid x,\, y_{<t})
= \arg\max_{w\in V} z_{t,w},
\]

where \(z_{t,w}\) are **logits** (monotone with softmax probabilities).

**Log-score of full string:**

\[
\log P_\theta(y_{1:n}\mid x)
= \sum_{t=1}^{n} \log P_\theta(y_t \mid x, y_{<t}).
\]

Greedy **maximizes each summand location-wise** assuming earlier tokens fixed—but that differs from maximizing the **sum** over **jointly chosen** \(y_{1:n}\).

---

## 4. Beam Search

Maintains \(B\) **partial hypotheses**. At each step, **expand** each hypothesis with **all** or **top-k** token extensions, **score**, then **prune** to top \(B\) partial sequences by cumulative metric.

**Unnormalized log-score:**

\[
S(y_{1:t}) = \sum_{i=1}^{t} \log P_\theta(y_i \mid x, y_{<i}).
\]

---

## 5. Length Normalization

Compare hypotheses of **different lengths** with:

\[
S_{\alpha}(y_{1:n}) = \frac{1}{n^{\alpha}} \sum_{t=1}^{n} \log P_\theta(y_t \mid x, y_{<t}).
\]

- \(\alpha=0\): no normalization (longer strings tend to have more **negative** raw log-probs).
- \(\alpha=1\): **per-token average logprob** (very common heuristic).
- \(\alpha \in (0,1)\): **compromise** penalizing brevity less than \(\alpha=1\).

**Rationale:** Product of conditional probabilities **shrinks** with length unless probabilities are nearly 1; normalization avoids pathological preference for **very short** outputs in beam ranking.

---

## 6. Why Greedy Is Suboptimal

Even with exact \(P_\theta\), **the joint MAP sequence** \(\arg\max_{y_{1:n}} P(y_{1:n}\mid x)\) **does not** reduce to per-step argmax unless the model satisfies strong **conditional independence** structures (which autoregressive LMs **do not**).

**Toy counterexample sketch:**

At \(t=1\), token \(a\) has highest **local** conditional prob, but all good **long** continuations after \(a\) are poor; a slightly lower first token \(b\) may open a branch where later tokens achieve **much higher** conditional probs such that the **total sum of logs** beats the greedy path.

---

## 7. Complexity Analysis

Let maximum length \(n\).

**Greedy:**

- Per step: \(\mathcal{O}(|V|)\) logits or \(\mathcal{O}(|V|)\) with top-k partial tricks in softmax.
- **Total:** \(\mathcal{O}(n\,|V|)\) time in naive form; in practice dominated by transformer forward at each step \(\mathcal{O}(n^2 d)\) with KV-cache in attention.

**Beam with full expansion:**

- Roughly **\(\mathcal{O}(B\cdot |V|\cdot n)\)** candidate scoring operations for naive **full-vocabulary** expansions each step before pruning—often too large—so implementations use **hard top-k** candidates per beam hypothesis.

**Memory:** \(\mathcal{O}(B\cdot n)\) to store tokens and scores.

**Statement requested by syllabus (aggregate form):** Beam search complexity is often summarized as **\(\mathcal{O}(B\times|V|\times n)\)** in **naive** full expansions; **optimized** stacks reduce effective \(|V|\) via pruning, GPU sampling kernels, or n-approx.

---

## 8. MAP View and Search

**Exact MAP:**

\[
y^\star = \arg\max_{y\in V^*} \; P_\theta(y\mid x)
= \arg\max_{y} \sum_{t} \log P_\theta(y_t\mid x,y_{<t}).
\]

**Search interpretation:**

- Greedy = **very shallow** search (width 1, myopic).
- Beam = **bounded-width** approximate combinatorial optimisation.
- For **log-linear** models with Markov structure, dynamic programming helps; for **general** autoregressive transformers, **no exact poly-time** MAP trick exists—beam is heuristic.

---

## 9. When Greedy Suffices vs When Beam Helps

| Scenario | Greedy often | Beam helps |
|----------|--------------|------------|
| Nearly **unimodal** next-token distributions | Good enough | Marginal |
| **Strong** local winner at each step correlates with global MAP | Good | Marginal |
| **Structured outputs** (code, math) with brittle early mistakes | May fail | Often improves |
| **Copying** long spans from input (pointer-like) | Can work with **constrained** decoding | Beam + constraints common |
| Open-ended creative generation | Quality may not correlate with prob | Sampling often preferred over beam |

**NMT history note:** RNN-era translation benefited from beam; with **instruction-tuned** chat models, many production stacks use **sampling** rather than beam for naturalness.

---

## 9A. Log-sum structure and why “local argmax” is not global MAP

Autoregressive factorization:

\[
\log P(y_{1:n}\mid x)
= \sum_{t=1}^{n} f_t(y_t),\quad f_t(y_t)=\log P(y_t\mid x,y_{<t}).
\]

Greedy chooses \(\hat{y}_t=\arg\max_{v} f_t(v)\) **sequentially**. A **global** MAP chooses \(\mathbf{y}^\star\in\arg\max_{\mathbf{y}\in V^n} \sum_t f_t(y_t)\).

**Observation:** This is a **discrete combinatorial** optimisation; only if choices **factorize independently** does sequential maximisation coincide—here \(f_t\) depends on the **whole** prefix \(y_{<t}\), yielding **path dependence**.

**Small theorem (prefix dominance is insufficient):** There exist SCMs where every strict prefix of \(\mathbf{y}^\star\) is **not** one-step greedy because an alternate first token lowers \(f_1\) but **raises** a long-run sum \(\sum f_t\). Beam search keeps such alternatives **alive** up to width \(B\).

**Connection to dynamic programming:** If each \(f_t\) depended only on a **finite state** of bounded memory (e.g., Markov order \(m\)), Viterbi-like DP would apply—Transformers have **unbounded** context dependence, so **exact** MAP is expensive.

---

## 10. Pseudocode

```
function greedy_generate(model, x, max_len):
    y ← empty list
    for t in 1..max_len:
        logits ← model(x, y)
        w ← argmax softmax(logits)
        append w to y
        if w == EOS: break
    return y

function beam_search(model, x, B, α, max_len):
    beams ← [(score=0, y=[])]
    for t in 1..max_len:
        candidates ← []
        for (S, y) in beams:
            logits ← model(x, y)
            # optionally restrict to top-K extensions per beam
            for w in vocabulary_or_topK(logits):
                Δ ← log P(w | x, y)  # from log_softmax
                S_new ← S + Δ
                candidates.append( (S_new, y + [w]) )
        prune candidates by S_new / (len(y)+1)^α   # length-normalised score
        keep top B hypotheses
        beams ← pruned
        if all EOS: break
    return argmax_final(beams by S / len^α)
```

---

## 10A. Worked micro-beam (illustrative)

Let \(B=2\), two-step generation, vocabulary \(\{A,B\}\). Suppose prefix is BOS.

| Step | Hypothesis | Last token | Stored \(\sum \log P\) (unnormalised) |
|------|------------|------------|---------------------------------------|
| 1 | h1 | A | \(\log 0.6\) |
| 1 | h2 | B | \(\log 0.4\) |
| 2 from h1 | AA | \(\log 0.6 + \log 0.2\) | \(\log 0.12\) |
| 2 from h1 | AB | \(\log 0.6 + \log 0.8\) | \(\log 0.48\) |
| 2 from h2 | BA | \(\log 0.4 + \log 0.85\) | **higher** if \(\log 0.34 > \log 0.12\) etc. |

Beam **keeps two best** rows at step 2; greedy might have locked into \(A\) at step 1 even if **BA** beats **AA** after two steps.

**Takeaway:** beam maintains **disjunction** of second-best prefixes; greedy commits **irreversibly** at \(t=1\).

---

## 11. Numerical Examples

### Example 1 — Two-step toy

Suppose after \(x\),

- Branch A: \(\log P(a|x)=-0.22\) (prob \(\approx 0.8\)), but next step best is poor: \(\log P(c|xa)=-1.61\). Total \(\log p \approx -1.83\).

- Branch B: \(\log P(b|x)=-1.00\) (prob \(\approx 0.37\)), but \(\log P(c|xb)=-0.10\). Total \(\approx -1.10\).

Greedy picks **a** first (higher local prob), yet **bc** beats **ac** globally under product of conditionals.

### Example 2 — Length normalization

Two completed strings:

- \(y^{(1)}\) length 4: total \(\log p=-8.0\). Average \(\frac{-8}{4}=-2.0\).

- \(y^{(2)}\) length 8: total \(\log p=-14.0\). Average \(\frac{-14}{8}=-1.75\).

**Raw sum** prefers \(y^{(1)}\) (\(-8 > -14\)), but **per-token average** prefers \(y^{(2)}\). Choosing \(\alpha\) tunes brevity bias.

---

## 12. Common Mistakes

- ❌ **Think argmax per step maximizes sequence joint probability.**

  ✓ It only maximizes a **nested greedy** approximation; beam or exact search may differ.

- ❌ **Compare beams of different lengths without normalization.**

  ✓ Apply \(\alpha\)-normalization or EOS-aware scoring.

- ❌ **Set beam \(B\) enormous for chat quality.**

  ✓ Large \(B\) can favor **generic high-probability** text; not always “better” subjectively.

- ❌ **Ignore logits numerics (float32 vs bfloat16).**

  ✓ Argmax stable; beams comparing **tiny** score differences can be noisy.

- ❌ **Assume beam complexity ignores top-k pruning.**

  ✓ Practical runtimes often **much better** than naive \(B|V|\) expansions.

---

## 13. Exercises

1. Construct a **3-level** binary tree where greedy path has **higher** product at depth 1 but **lower** product at full depth.

2. Prove \(\arg\max_y \prod_t p_t(y_t)\) with factors \(p_t(y_t)=P(y_t\mid\cdot)\) **is not** solved by per-step argmax—give explicit \(2\times 2\) table counterexample.

3. Show how \(\alpha\) affects preference for length \(n\) when per-token logprobs are i.i.d. negative constants (pathological sanity check).

4. Relate beam search to **Viterbi** on **HMM**—what structural difference breaks exact DP for Transformers?

5. Estimate time per token given \(T_\mathrm{fwd}\) for one forward pass—how does doubling \(B\) change steps if using **parallel** batch expansions?

---

### Footnote

Decoding is **inference-time** algorithmics on a **fixed** model; changing decoding **does not** change likelihoods— it changes which mode of \(P_\theta\) you **surface**.
