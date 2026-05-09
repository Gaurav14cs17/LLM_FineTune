# Understanding SentencePiece Unigram Language Model Tokenization

*A unigram lexicon, EM re-estimation, Viterbi segmentation, and contrast with greedy BPE*

---

**SentencePiece** (Kudo & Richardson, 2018) ships **language-agnostic** tokenisers that operate on **raw** Unicode text (often as **escaped bytes**) avoiding hand-crafted pre-tokenisation. Its **Unigram Language Model (ULM)** mode starts from a **large** seed vocabulary and **prunes** token types by an **EM-style** procedure maximising the marginal likelihood of the training corpus under a **mixture over segmentations**. Decoding uses **Viterbi** shortest-path on a **DAG** of permitted substrings—**globally** optimal under fixed probabilities—not the left-to-right **greedy** growth of BPE.

**Reading map:** probabilities → EM pruning → Viterbi decode; contrast with **BPE**’s rule stack wherever you see “global vs greedy.”

---

## Table of Contents

1. [Overview](#1-overview)
2. [Generative Story for a Single String](#2-generative-story-for-a-single-string)
3. [Parameterisation: Unigram LM](#3-parameterisation-unigram-lm)
4. [EM Algorithm: E-step and M-step](#4-em-algorithm-e-step-and-m-step)
   - [Forward-Backward on the Segmentation DAG (Sketch)](#forward-backward-on-the-segmentation-dag-sketch)
   - [Stochastic Segmentation (Training-Time Augmentation)](#stochastic-segmentation-training-time-augmentation)
5. [Vocabulary Pruning Under marginal Δ Loss](#5-vocabulary-pruning-under-marginal-δ-loss)
6. [Decoding: Viterbi for \(\arg\max\) Segmentation](#6-decoding-viterbi-for-argmax-segmentation)
7. [Loss Decomposition Sketch](#7-loss-decomposition-sketch)
   - [7.1 EM Monotonicity (Standard Result)](#71-em-monotonicity-standard-result)
8. [Comparison — BPE (Greedy) vs Unigram (Probabilistic)](#8-comparison--bpe-greedy-vs-unigram-probabilistic)
9. [SentencePiece Surface: `▁` and Lossless Reconstruction](#9-sentencepiece-surface--and-lossless-reconstruction)
   - [9.1 Bytes, the SentencePiece space marker, and OOV symmetry](#91-bytes-the-sentencepiece-space-marker-and-oov-symmetry)
10. [Pseudocode](#10-pseudocode)
11. [Common Mistakes](#11-common-mistakes)
12. [Exercises](#12-exercises)

---

## 1. Overview

### VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(x\) | Input string (SentencePiece: may include spaces via `▁`) |
| \(S(x)\) | A **segmentation** of \(x\) into pieces \(x_1^k = (x_1,\ldots,x_k)\) |
| \(V\) | Subword vocabulary (types) with **probability** masses \(p(t)\), \(t\in V\) |
| \(\mathcal{S}(x)\) | Set of **feasible** segmentations where each piece \(\in V\) |
| \(L\) | Corpus log-likelihood / description length objective |
| **E-step** | Infer distribution over segmentations under current \(p\) |
| **M-step** | Update \(p(t)\) from expected counts |
| **Δ** | Approximate log-likelihood change when removing symbol \(t\) |

### ┌─────────────────────────────────────────────────────────────┐
### │ INTUITION: many segmentations, one distribution             │
### └─────────────────────────────────────────────────────────────┘

```
BPE (greedy grow):
  aa bb cc  → merges driven by *local* pair stats (fast, myopic)

Unigram LM:
  "newest" can split as:
      newest | new+est | n+ewest | ...  (all legal if pieces exist)
  EM distributes probability mass across *alternatives* so global p(t)
  aligns with how often pieces appear in *likely* parses.

        posterior over segmentations
                 ┌───────────────┐
   EM  ◄────────►│ refresh p(t) │
                 └───────────────┘
   until pruning removes redundant pieces with tiny marginal payoff
```

---

## 2. Generative Story for a Single String

1. **Choose** a segmentation \(S(x) \in \mathcal{S}(x)\) with probability \(q(S \mid x)\) (the **E-step** defines this as the **posterior** under current parameters).
2. **Emit** each consecutive piece \(x_i\) **independently** from a categorical \(\mathrm{Unigram}(p)\):

\[
P(x, S) = \mathbb{1}[S \in \mathcal{S}(x)] \cdot \prod_{i=1}^{k} p(x_i).
\]

Marginalising over \(S\):

\[
P(x) = \sum_{S \in \mathcal{S}(x)} P(S \mid x)\, \prod_{i=1}^{|S|} p(x_i) \quad\text{(conceptually)}.
\]

In EM we never need \(P(x)\) **explicitly** for all \(x\) in closed form on huge corpora; we accumulate **expected counts**.

---

## 3. Parameterisation: Unigram LM

Let \(\theta_t\) be **unnormalised** scores; **softmax** yields probabilities:

\[
p_\theta(t) = \frac{\exp(\theta_t)}{\sum_{u \in V} \exp(\theta_u)}.
\]

**Constraint:** \(\sum_{t\in V} p(t) = 1\) (some implementations keep a tiny **unk / fallback** bucket or rely on **byte** base set for coverage).

**Key property:** a **fatter** vocabulary allows **shorter** codes but risks **overfitting**; pruning trades off fit vs capacity.

---

## 4. EM Algorithm: E-step and M-step

### E-step (expected sufficient statistics)

For each training string \(x\), enumerate (implicitly via DP) **all** segmentations compatible with \(V\). Compute **posterior** weights \(q(S\mid x)\) **proportional** to \(\prod_{t\in S} p(t)\). Then **expected count** of token \(u\):

\[
\tilde{c}(u) = \sum_{x \in \text{corpus}} \sum_{S \in \mathcal{S}(x)} q(S\mid x)\, \underbrace{\#\text{ times } u \text{ appears as a piece in } S}_{c(u;S)}.
\]

**Forward–backward** on a **lattice** computes these expectations **without** enumerating all \(S\) explicitly.

### M-step (multinomial update)

With \(\lambda\) smoothing optional,

\[
p^{\mathrm{new}}(u) = \frac{\tilde{c}(u)}{\sum_{t \in V} \tilde{c}(t)}.
\]

Iterate until **LL increases** marginally. This is **standard EM** for a **mixture** where latent variable is segmentation.

### Forward-Backward on the Segmentation DAG (Sketch)

Index string positions \(0..|x|\). Build directed edges \((i \to j)\) whenever substring \(x[i:j)\in V\) with weight \(w_{i,j} = \log p(x[i:j])\). Every full segmentation corresponds to a **path** \(0 \to |x|\).

**Forward messages** (log-semiring or linear domain; linear shown for clarity): \(\alpha(j)\) sums probabilities of all paths **ending** at \(j\):

\[
\alpha(j) = \sum_{i<j,\ x[i:j]\in V} \alpha(i)\, p(x[i:j]),\quad \alpha(0)=1.
\]

**Backward messages:** \(\beta(i)\) sums probabilities of completing \(x\) from position \(i\):

\[
\beta(i) = \sum_{j>i,\ x[i:j]\in V} p(x[i:j])\, \beta(j), \quad \beta(|x|)=1.
\]

**Edge marginal** (probability that optimal **mixture** uses edge \(i\to j\) across all paths, weighted by likelihood):

\[
\gamma(i,j) \;\propto\; \alpha(i)\, p(x[i:j])\, \beta(j).
\]

Normalise \(\sum_{i,j}\gamma(i,j)\) along each **corpus example** to obtain **expected counts** \(\tilde c(u)\): accumulate \(\gamma\) over all edges labelled with substring \(u\).

**Complexity:** \(O(|x|^2)\) naive edge loop; **Trie**-filtered edges yield **near-linear** scaling in \(|x|\) for compact vocabularies over byte alphabets.

### Stochastic Segmentation (Training-Time Augmentation)

Some setups **sample** segmentations from \(q(S\mid x)\) instead of always MAP—**subword regularisation** reduces boundary overfitting; SentencePiece exposes related options in certain training modes.

---

## 5. Vocabulary Pruning Under marginal Δ Loss

Start \(|V|\) **large** (frequent n-grams/bytes). **While** \(|V| > V_{\text{target}}\):

1. Compute **current** LL under EM-fixed \(p\): \(\mathcal{L}(V)\).
2. For each candidate symbol \(t \in V\), estimate \(\Delta_t \approx \mathcal{L}(V) - \mathcal{L}(V\setminus\{t\})\) via **re-running** a cheap surrogate or the paper’s approximation (implementation uses **loss change** from forced removal in Viterbi parses).
3. **Drop** bottom **percentage** (e.g. 10–20%) of symbols with **smallest** \(\Delta_t\) (least marginal contribution).
4. **Re-EM** until stable; repeat.

**Intuition:** symbols carrying **unique** shortest paths for many tokens incur **large** \(\Delta_t\) when removed; symbols **substitutable** by alternate splits have **small** \(\Delta_t\).

---

## 6. Decoding: Viterbi for \(\arg\max\) Segmentation

For fixed \(p\), **MAP** segmentation minimises **code length** \(-\sum \log p(x_i)\):

\[
S^\*(x) = \arg\max_{S \in \mathcal{S}(x)} \sum_{x_i \in S} \log p(x_i).
\]

Dynamic programming:

\[
\mathrm{best}[j] = \max_{0\le i<j,\ x[i:j]\in V} \ \mathrm{best}[i] + \log p(x[i:j]),
\]

with \(\mathrm{best}[0]=0\). **Backpointers** recover the split.

**Complexity:** \(O(|x| \cdot |V|_{\text{local}})\) with trie lookups to filter edges.

---

## 7. Loss Function Derivation Sketch

**Corpus negative log-likelihood** (summed over observations \(x\) with empirical weights \(w_x\)):

\[
\mathcal{L}(\theta) = -\sum_x w_x \log \Bigl(\sum_{S \in \mathcal{S}(x)} \prod_{t \in S} p_\theta(t)\Bigr).
\]

**EM lower bound** at iteration \(r\):

\[
\mathcal{Q}(q,\theta) = \sum_x w_x \sum_{S} q(S\mid x)\, \Bigl[ \sum_{t \in S} \log p_\theta(t) - \log q(S\mid x) \Bigr].
\]

Maximising \(\mathcal{Q}\) w.r.t. \(\theta\) yields the **multinomial** M-step; choosing \(q\) as posterior tightens the bound (**EM guaranteed** to improve actual \(\mathcal{L}\) under exact E-step, mild conditions).

**Pruning heuristic** uses approximate \(\mathcal{L}(V) - \mathcal{L}(V\setminus\{u\})\) as a **discrete** derivative measuring **sensitivity** of the marginal likelihood to token \(u\).

### 7.1 EM Monotonicity (Standard Result)

Each EM iteration **raises** the marginal log-likelihood \(\mathcal{L}(\theta)\) **unless** already at a fixed point (under exact E-step). **Pruning**, however, is a **discrete** non-gradient operation—it can **temporarily** drop \(\mathcal{L}\) if \(\Delta\) estimates are noisy; practitioners **re-run** EM after each pruning wave to **repair** likelihood.

---

## 8. Comparison — BPE (Greedy) vs Unigram (Probabilistic)

| Criterion | BPE | Unigram LM (SentencePiece) |
|-----------|-----|----------------------------|
| Search topology | grows merges **monotone** | prunes from **large** vocab |
| Optimality | local **pair** choice | **global** objective under EM + pruning tiers |
| Segmentation at inference | **rule stack** outcomes | **Viterbi** |
| Failure modes | frequent raw n-grams dominate early | requires **careful** seed + smoothing |
| Speed | very fast | heavier (EM loops) |

---

## 9. SentencePiece Surface: `▁` and Lossless Reconstruction

SentencePiece encodes **spaces** with **`▁` (U+2581)** prepended to the **first** subword of each original **space-delimited** word in the surface string:

```
Raw:    "Hello world"
Pieces might be: [▁Hello, ▁world] or similar depending on vocab
```

**Reconstruction** concatenates pieces and translates `▁` → space (**language neutral**, no English-specific regex split).

### 9.1 Bytes, the SentencePiece space marker, and OOV symmetry

Because SentencePiece can train on **escaped UTF-8 bytes**, **every** surface string has **some** decomposition into base atoms—**true OOV at byte level is empty**. The interesting failures shift to **pathological probabilities** (extremely low-probability segmentations) rather than **hard unknowns**. Models should still **regularise** rare byte n-grams via **pruning** and **smoothing** so Viterbi does not stitch paths through **nearly-zero** probability edges that **blow up** NLL.

**Practical upshot:** monitoring **median** piece length plus **histogram** of \(-\log p(\text{piece})\) catches silent pathology earlier than classic `<unk>` metrics.

Keep a held-out **sentence** shard to verify reconstruction round-trips **byte-identical** to inputs when lossless mode is enabled.

### NUMERICAL MICRO-EXAMPLE (toy probabilities)

Let allowed pieces of `"aa"` be \(\{a,p=0.6\}\) and \(\{aa,p=0.4\}\) with **exclusive** choices (toy **two** parses):

- Parse \([a,a]\): \(\log 0.36\).
- Parse \([aa]\): \(\log 0.4\).

**Viterbi** picks \([aa]\). EM would **rebalance** \(p(a), p(aa)\) if corpora repeatedly prefer one parse’s evidence.

---

## 10. Pseudocode

```
# Viterbi MAP segmentation for a string x[0..n) given vocab V with log p[t]

def viterbi_best(x, V, logp):
    n = len(x)
    best = [-inf]*(n+1); back = [None]*(n+1)
    best[0] = 0.0
    for j in range(1, n+1):
        for i in range(0, j):
            sub = x[i:j]
            if sub in V:
                cand = best[i] + logp[sub]
                if cand > best[j]:
                    best[j] = cand; back[j] = i
    # recover by backpointers
    return decode_pieces(back, x)

# EM for unigram LM with lattice expectations uses forward-backward on DAG edges
# Prune bottom-k% tokens by approximate Δ loss, re-EM, repeat until |V| target
```

---

## 11. Common Mistakes

- ❌ **Assuming Viterbi EM fully Bayesian** — EM is **point** estimate; no posterior over \(\theta\). ✓ Use **held-out** checks for **pruning** decisions.
- ❌ **Forgetting normalisation** after each M-step — probabilities must **sum to 1** over \(V\) (plus any explicit **fallback** bucket). ✓ Verify \(\sum_t p(t)\approx 1\).
- ❌ **Equating SentencePiece Unigram with BPE shipped in SentencePiece** — the **same** library implements **both**; training recipes differ. ✓ Check **`model_type`** flag.
- ❌ **Ignoring byte fallbacks** — without a closed **character/byte** base, **OOV** surfaces break. ✓ Always include **byte-level** atoms or `<unk>` handling.

---

## 12. Exercises

1. **Lattice DP:** derive recurrence for `best[j]` including **backpointers**; prove \(O(n^2)\) edge checks reduce with **Trie** of \(V\) to \(O(n \cdot \text{max token length})\) in typical alphabets (explain trie lookup bounds).
2. **EM proof:** write the **ELBO** for mixture-of-segmentations and show the M-step multinomial solution **closed form**.
3. **Compare** greedy BPE decoding vs Viterbi under a **hand-designed** \(p\) on a **three-letter** string where two **distinct** segmentations tie—does MAP **break ties** arbitrarily? Discuss **determinism** policy.
4. **Pruning:** explain why removing a token that participates in **all** MAP segmentations of a frequent word spikes \(\Delta\) — estimate a **first-order** approximation using corpus counts.

---

*SentencePiece Unigram LM marries a **simple generative** story (independent piece emissions) with **combinatorial** latent structure (segmentation), making it the probabilistic counterweight to **greedy** subword heuristics.*
