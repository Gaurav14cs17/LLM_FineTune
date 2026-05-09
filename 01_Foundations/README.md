# Chapter 01 — Mathematical Foundations

Language models are **probability models**. Before attention graphs or tokenizer byte-pair rules, we pin down what \(P_\theta\) **means**: a conditional product measure over token sequences, trained by minimising an empirical cross-entropy whose minima link directly to coding length and to Kullback–Leibler gaps from the true(but unknown) generating process. This chapter is deliberately **symbol-heavy**: later material imports these results as black boxes unless the foundations are solid.

```
     UNKNOWN DATA GENERATING PROCESS P* over Unicode / tokens
                           │
                           ▼
              empirical corpus  ≈  finite sample from P*
              (often treated as i.i.d. 'documents' — a modeling choice)
                           │
                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Factorisation:  P_θ(w_1,…,w_T) = ∏_t P_θ(w_t | w_<t)        │
        │                                                              │
        │  LOSS:  L(θ) = - (1/T) ∑_t log P_θ(w_t | w_<t)   [nats/token] │
        │            exp(L) = perplexity  (effective branch factor)      │
        └──────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴───────────────────┐
        │  H(P), H(P,Q), KL(P‖Q)              │
        │  relate "bits per token" to fit     │
        └─────────────────────────────────────┘
```

## Sub-chapters

**`01_Probability_Basics/`**  
Derives the chain rule for sequences and manipulates conditional objects \(P(A\mid B)\), including Bayes’ rule when we discuss noisy supervision in later alignment chapters. Establishes what independence would mean for tokens (usually **false**) and why the autoregressive LM remains tractable **without** independence.

**`02_Language_Model_Definition/`**  
Fixes notation for token indices, context windows, and corpus-level versus token-level loss averages. Defines perplexity \(\mathrm{PPL}=\exp(\mathrm{NLL})\) and interprets it as the geometric mean of the inverse probabilities of the actual next tokens. Sets the stage for comparing models that differ only in tokenisation (same text, different segmentations).

**`03_Entropy_and_Information/`**  
Defines entropy \(H(P)= -\sum_x P(x)\log P(x)\), cross-entropy \(H(P,Q)\), and \(\mathrm{KL}(P\|Q)\). Uses Gibbs’ inequality \(\mathrm{KL}(P\|Q)\ge 0\) to show **why** minimising NLL under empirical \(P\) is equivalent to minimising \(H(P_\mathrm{emp},Q_\theta)\), and connects the gap \(\mathrm{KL}(P^\star\|P_\theta)\) to irreducible error in scaling-law discussions.

## Key formulas and concepts (chapter-level)

**Autoregressive factorisation**

\[
P_\theta(w_1,\ldots,w_T) = \prod_{t=1}^{T} P_\theta(w_t \mid w_1,\ldots,w_{t-1}).
\]

**Average negative log-likelihood (one common definition)**

\[
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(w_t\mid w_{<t}).
\]

**Perplexity**

\[
\mathrm{PPL}(\theta) = \exp\bigl(\mathcal{L}(\theta)\bigr) = \exp\Bigl(-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(w_t\mid w_{<t})\Bigr).
\]

**Cross-entropy and KL (discrete)**

\[
H(P,Q) = -\sum_x P(x)\log Q(x), \qquad \mathrm{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}.
\]

**Decomposition used everywhere**

\[
H(P,Q) = H(P) + \mathrm{KL}(P\|Q).
\]

**Interpretation:** \(\mathcal{L}(\theta)\) is the sample estimate of \(H(P_{\mathrm{emp}},P_\theta)\); lowering it reduces the KL term toward the **empirical** distribution **unless** regularisation or mismatched tokenisation skews \(P_{\mathrm{emp}}\).

## Prerequisites

- Discrete probability: events, random variables, expectation (at least finitely supported variables).
- Log and exponential identities; comfort reading \(\log\) as \(\ln\) (nats) in ML.
- Optional: very basic optimisation intuition (gradients appear starting in Chapter 06).

## Positioning in the curriculum

| Downstream chapter | Uses from Foundations |
|--------------------|------------------------|
| 02 Tokenization | Support of random variable (tokens) changes; compare perplexities fairly |
| 03 Embeddings | Continuous representations parameterise softmax over **same** support |
| 06 Pretraining | Objective is still \(\mathcal{L}\); AdamW minimises noisy estimates of it |
| 09 Scaling laws | Irreducible loss \(E\) connects to \(H(P^\star)\) plus approximation error |

## Study tips

- Re-derive \(\mathrm{KL}\ge 0\) from Jensen’s inequality once by hand; you will recognise the same convexity in other losses (DPO, RLHF surrogate objectives) later.
- Track **nats vs bits** consciously (\(\log_2\) vs \(\ln\)); papers mix conventions; perplexity is invariant up to the base change in the **exponent**.

## Quick reference (mnemonics)

- **Likelihood** increases when assigned probability to observed tokens increases; **NLL** decreases.
- **KL** is asymmetric: training cares about \(\mathrm{KL}(P_\mathrm{emp}\|P_\theta)\) via cross-entropy, not the reverse, under the usual empirical risk setup.
- **PPL 100** means roughly a net **1%** chance for the right token on average **only** if tokenisation is fixed; cross-model PPL comparison needs **same tokenizer / same test protocol**.

## Mini glossary

| Term | One line |
|------|----------|
| NLL | Average \(-\log p\) of true next tokens under the model |
| Calibrated vs accurate | Low NLL implies good **ranking** of alternatives on average; separate calibration matters for **absolute** probabilities |
| Empirical distribution | Frequency counts from finite training text (smoothed in some older LMs) |

---

_End of chapter overview._

