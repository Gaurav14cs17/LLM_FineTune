# Chapter 01 — Mathematical Foundations

Everything downstream (loss functions, attention, training stability) is built on a handful of
probability and information-theory identities. This chapter derives them carefully.

---

## Sub-topics

| # | Folder | Core idea |
|---|--------|-----------|
| 1 | `01_Probability_Basics/` | Chain rule, conditional probability, Bayes |
| 2 | `02_Language_Model_Definition/` | NLL loss, perplexity |
| 3 | `03_Entropy_and_Information/` | Entropy, cross-entropy, KL divergence |

---

## Key Equations at a Glance

**Chain Rule of Probability**
```
P(w₁, w₂, …, wₙ) = ∏_{t=1}^{n} P(wₜ | w₁, …, wₜ₋₁)
```

**Negative Log-Likelihood (NLL)**
```
L(θ) = − (1/T) ∑_{t=1}^{T} log P_θ(wₜ | w₁:ₜ₋₁)
```

**Perplexity**
```
PPL = exp( L(θ) ) = exp(−(1/T) ∑_t log P_θ(wₜ|context))
```

**Shannon Entropy**
```
H(X) = −∑_{x} P(x) log₂ P(x)      [bits]
```

**Cross-Entropy Loss** (what models actually minimise)
```
H(p, q) = −∑_{x} p(x) log q(x)   ≥ H(p)
```

**KL Divergence** (information gap between distributions)
```
KL(p ‖ q) = ∑_x p(x) log [p(x)/q(x)]  ≥ 0   (by Jensen)
```

Relation: `H(p,q) = H(p) + KL(p‖q)`
