# Speculative Decoding for Autoregressive LLMs

Accelerate **exact** decoding from a **target** distribution \(P_{\mathrm{target}}\) using a **cheap draft** model \(P_{\mathrm{draft}}\). Core idea: propose **\(K\)** candidate suffix tokens per round, then **verify** with **parallel forward passes** on the target. Covers **rejection sampling correction**, **proof of exactness**, **expected speedup**, and failure modes.

---

## Table of Contents

- [1. Variables](#1-variables)
- [2. Intuition — Amortize Target Forward Passes](#2-intuition--amortize-target-forward-passes)
- [3. Setup: Draft vs Target](#3-setup-draft-vs-target)
- [4. One-Round Algorithm (Draft then Verify)](#4-one-round-algorithm-draft-then-verify)
- [5. Acceptance Rule and Rejection Sampling](#5-acceptance-rule-and-rejection-sampling)
- [6. Proof Sketch: Output Distribution Matches Target](#6-proof-sketch-output-distribution-matches-target)
  - [6.1 Correctness via composition (finite vocabulary)](#61-correctness-via-composition-finite-vocabulary)
- [7. Expected Speedup Formula](#7-expected-speedup-formula)
  - [7.1 Geometric model for consecutive accepts (toy)](#71-geometric-model-for-consecutive-accepts-toy)
- [8. Parallel Verification — One Forward for Many Positions](#8-parallel-verification--one-forward-for-many-positions)
- [9. Pseudocode (Full Round)](#9-pseudocode-full-round)
- [10. Numerical Examples](#10-numerical-examples)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)

---

## 1. Variables

| Symbol | Meaning |
|--------|---------|
| \(P_{\mathrm{target}}\) | Large **oracle** model distribution at each position |
| \(P_{\mathrm{draft}}\) | Small draft model providing fast proposals |
| \(K\) | Number of speculative candidate tokens per round (aka \(\gamma\) in some papers) |
| \(x_{:t}\) | Context prefix accepted up to time \(t\) |
| \(u\) | Proposed token at a verification substep |
| \(a\) | Acceptance probability in rejection sampling step |
| \(q\) | Draft probability at verification (**same distribution** used when sampling draft) |
| \(p\) | Target probability (**ideal** distribution we want to sample exactly) |
| \(S\) | Expected accepted tokens per target-heavy phase |
| \(\mathrm{speedup}\) | Tokens per wall-clock vs baseline AR decoding |

---

## 2. Intuition — Amortize Target Forward Passes

```
Standard AR (target only):
   t ──► t+1 ──► t+2 ──► ...     one BIG forward each step (serial)

Speculative (draft + target):
   draft: small model proposes  u1, u2, ..., uK   (cheap serial)
   target: ONE forward evaluates logits at stacked positions in parallel
           verify / accept prefix of {u1,...,uK}
```

**Key:** GPUs often underutilized at small batch—**batched** target evaluation across positions can multiply throughput if **acceptance** rate is decent.

---

## 3. Setup: Draft vs Target

Let the **desired sampling law** be **exactly** the target model:

\[
\text{Want } y_{t+1:t+K} \sim \prod_{i=1}^{K} P_{\mathrm{target}}(\,\cdot \mid x, y_{:t+i-1}).
\]

The draft proposes **likely** continuations **under its own** dynamics:

\[
\tilde{y}_{t+i} \sim P_{\mathrm{draft}}(\,\cdot \mid x,\tilde{y}_{:t+i-1}).
\]

Verification compares **draft vs target** probabilities to perform a statistically correct **accept/reject** or **max coupling** style step—implementation variants exist, but the **mathematical contract** is **restore exactness**.

---

## 4. One-Round Algorithm (Draft then Verify)

For each speculative round:

1. **Draft:** autoregressively sample \(K\) tokens \(\tilde{y}_{t+1:t+K}\) from \(P_{\mathrm{draft}}\) conditioned on accepted prefix.

2. **Verify:** using **target** model, obtain conditional probabilities \(p_i(v)=P_{\mathrm{target}}(v\mid \text{prefix})\) for positions \(i=1..K\) in **one** parallel forward (causal masking).

3. **Sequential accept test** along \(i=1..K\):

   - Compare \(p_i(\tilde{y}_{t+i})\) and \(q_i(\tilde{y}_{t+i})\) where \(q_i\) is **draft's** probability for the **same** token at that prefix when draft produced it.

   - Accept token \(i\) with probability min\(\bigl(1,\frac{p_i(\tilde{y}_{t+i})}{q_i(\tilde{y}_{t+i})}\bigr)\) **(Metropolis-style acceptance / rejection sampling ratio bound)**; exact details align with the chosen speculative algorithm family (Leviathan / **speculative decoding** canonical formulation).

4. On first rejection at position \(i\), **resample** a corrected token from the residual distribution so the **marginal** matches \(p_i(\cdot)\).

---

## 5. Acceptance Rule and Rejection Sampling

**Rejection sampling background:** To sample from \(p\) but only know how to sample \(q\), draw \(y\sim q\) and accept with

\[
a(y) = \min\!\left(1, \frac{p(y)}{M\,q(y)}\right),
\]

where \(M\) covers \(p/q\) if bounded. In LLM stacks, ratios use **matching prefixes** and **same** token \(y\).

**Per-step acceptance (canonical form used in speculative decoding writeups):**

Define \(r_i = \dfrac{p_i(\tilde{y}_{t+i})}{q_i(\tilde{y}_{t+i})}\). Accept \(\tilde{y}_{t+i}\) w.p. \(\min(1,r_i)\).

If rejected, sample replacement from distribution

\[
\frac{\max(0,\, p_i(v) - q_i(v))}{Z}
\quad\text{(residual mass after draft overlap)},
\]

up to normalization constants encoded in official references—this **repairs** the coupling so marginals equal \(p_i\).

---

## 6. Proof Sketch: Output Distribution Matches Target

**Invariant per position \(i\)** (conditioned on prior accepted prefix **exact** under target):

- Draft proposes \(\tilde{y}\sim q_i\).
- Accept–reject with \(\min(1,\frac{p_i(\tilde{y})}{q_i(\tilde{y})})\) yields **accepted** samples following \(p_i\) **(rejection sampling theorem)**.

If rejected, **correction sampling** from normalized \(\max(0,p_i-q_i)\) (or equivalent **max coupling** sampling) ensures the **next accepted token** is exactly \(p_i\).

By induction along time steps, the **stream of accepted tokens** is **identically distributed** to **naive autoregressive sampling** from \(P_{\mathrm{target}}\).

### 6.1 Correctness via composition (finite vocabulary)

Let \(p,q\) be distributions on finite \(V\). **Rejection sampling** draws \(Y\sim q\), accepts with probability \(r(Y)=\min\bigl(1,\frac{p(Y)}{M\,q(Y)}\bigr)\) for envelope \(M\ge \max_y \frac{p(y)}{q(y)}\) (with convention \(0/0=0\)). The probability of accepting **and** outputting \(y\) equals \(\min\bigl(q(y),\frac{p(y)}{M}\bigr)\). Conditioning on acceptance yields \(p\) **exactly** as the output law. Speculative decoding realises an analogous **per-position** contract with **shared prefixes**; the **algorithmic** nuance is bookkeeping \(q\) as the **draft’s** conditional for the **very** token proposed along a **single** trajectory.

**ASCII coupling:**

```
     q  proposes ──► compare p vs q on SAME token
                         │
                         ├── accept path  (exact p marginal)
                         │
                         └── reject path  (correction draw)  (exact p marginal)
```

---

## 7. Expected Speedup Formula

Let:

- \(c_{\mathrm{draft}}\) = time for draft to produce \(K\) tokens,
- \(c_{\mathrm{verify}}\) = time for one **batched** target forward across up to \(K\) positions.

**Ballpark per-new-token wall-clock under ideal parallelism:**

Let \(\mathbb{E}[S]\) be expected **accepted tokens per round**.

**Speedup vs baseline** target-only per token \(\approx\)

\[
\frac{c_{\mathrm{target\_step}}}{\dfrac{c_{\mathrm{draft}} + c_{\mathrm{verify}}}{\mathbb{E}[S]}}.
\]

Useful **intuition:** If one verification can validate **many** draft tokens (\(S\) near \(K\)), amortized target cost drops nearly **\(K\times\)** per accepted token—if draft is **cheap** and alignment is **high**.

A common textbook-style factorization:

\[
\mathrm{Speedup} \approx
\frac{1}{
\frac{c_{\mathrm{draft}}}{K\cdot c_{\mathrm{target\,ref}}}}
+ \frac{c_{\mathrm{verify}}}{K\cdot c_{\mathrm{target\,ref}}}
}
\cdot \mathbb{E}[S].
\]

(Exact constants depend on kernel batching, KV-cache sizes, \(K\), implementation.)

**Acceptance probability:** Heuristically grows when **draft ≈ target** locally (student–teacher agreement) on high-confidence tokens.

### 7.1 Geometric model for consecutive accepts (toy)

Assume **independent** identical acceptance probability \(\alpha\in(0,1]\) across slots (approximation only). Let \(S\) be the number of accepted draft tokens before first rejection or horizon \(K\):

\[
\mathbb{P}(S=s)=\alpha^{s}(1-\alpha)\ (s<K),\qquad \mathbb{P}(S=K)=\alpha^{K}.
\]

Then

\[
\mathbb{E}[S]=(1-\alpha)\sum_{s=0}^{K-1} s\alpha^{s} + K\alpha^{K}.
\]

For \(\alpha\to 1\), \(\mathbb{E}[S]\to K\); for small \(\alpha\), increasing \(K\) yields **diminishing returns** because rounds terminate early—motivates **adaptive** \(K\).

---

## 8. Parallel Verification — One Forward for Many Positions

Given accepted prefix \(x_{:t}\) and candidate extension \(\tilde{y}_{t+1:t+K}\), transformer with **causal attention** can compute **next-token distributions** for each prefix position needed in **one** forward on a window **if** logits for **each prefix state** are exposed via **offset** prefill techniques.

**Complexity note:** You still pay \(\mathcal{O}((t+K)^2)\) attention in naive implementations—**KV caching** and careful scheduling make the dominant cost **near-linear** in practical lengths for moderate \(K\).

---

## 9. Pseudocode (Full Round)

```
function speculative_round(x_prefix, K):
    # Draft phase (often on CPU / small GPU)
    draft_tokens ← []
    for i in 1..K:
        u ~ P_draft(· | x_prefix ◦ draft_tokens)
        append u to draft_tokens
        store q[i] ← P_draft(u | appropriate prefix)

    # Verify phase (large model, parallel logits for needed positions)
    for i in 1..K:
        p[i] ← P_target(· | x_prefix ◦ draft_tokens[1:i-1])  # parallel in optimized kernels

    accepted ← 0
    for i in 1..K:
        r ← p[i](draft_tokens[i]) / q[i]        # guard numerical floors
        if Bernoulli(min(1, r)):
            accepted += 1
            extend x_prefix with draft_tokens[i]
        else:
            y ← sample_correction_from_target_residual(p[i], q[i])
            extend x_prefix with y
            break   # end round early after resample

    if accepted == K:
        # optional extra sampling for next start; implementations vary
        return x_prefix
    return x_prefix
```

**Note:** Exact **break/continue** rules match the reference implementation you follow—**mathematics**: preserve per-position **exactness** after correction.

---

## 10. Numerical Examples

### Example 1 — Acceptance ratio

Suppose at some step, for proposed token \(u\),

\[
p(u)=0.12,\qquad q(u)=0.20.
\]

Acceptance probability \(= \min(1,\frac{0.12}{0.20}) = 0.6\).

Interpretation: target finds the draft’s token **plausible but not** overly tilted vs draft—**40%** need **resample** from residual.

### Example 2 — Perfect alignment

If **everywhere** \(q=p\), acceptance is **1** and no corrections ⇒ maximal speedup (idealized **distillation match**).

### Example 3 — Speedup back-of-envelope

Suppose baseline target takes **10** ms/token. A round with \(K=4\) spends **2** ms draft + **12** ms batched target verify and accepts **3.2** tokens on average.

**Amortized target-equivalent**\(\approx (2+12)/3.2=4.375\) ms/token **if** dominated by these phases \(\Rightarrow\) speedup \(\approx 10/4.375 \approx 2.3\times\).

(Toy numbers only.)

---

## 11. Common Mistakes

- ❌ **Assume speculative decoding changes \(P_{\mathrm{target}}\).**

  ✓ Correct implementation samples **exactly** from target (up to floating error).

- ❌ **Use mismatched \(q\) (different temperature than draft sampling).**

  ✓ Store **the exact logprobs** used when sampling each proposal.

- ❌ **Skip correction after rejection.**

  ✓ Otherwise marginals deviate—**biased** outputs.

- ❌ **Set \(K\) huge without measuring acceptance.**

  ✓ High \(K\) may waste verify FLOPs if reject early—**adaptive \(K\)** exists.

- ❌ **Ignore memory bandwidth** of big target forward at long contexts.

  ✓ Speedups shrink when **memory-bound**.

---

## 12. Exercises

1. **Prove** standard rejection sampling: accepted draws follow \(p\) when proposals come from \(q\) with acceptance \(\min(1, p/(Mq))\).

2. Write the **total variation distance** bound between approximate speculative sampling without corrections vs true \(p\).

3. If **\(p\le q\) pointwise** always, show acceptance is **always 1**. What does that imply about draft quality?

4. **Derive** expectation of accepted length in **geometric** model when per-step accept prob constant \(=\alpha\).

5. Explain why **speculative execution** on **CPU** branch predictors is **not** mathematically exact—what differs for LLMs?

---

### Coda

Speculative decoding trades **algorithmic bookkeeping** for **GPU wavefront utilization**. Pair it with **draft model distillation** from target for maximal acceptance; measure **tokens/sec** and **exact-match distribution tests** vs brute-force AR on short sequences.
