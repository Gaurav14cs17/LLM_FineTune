# Understanding Entropy, Cross-Entropy, and KL Divergence

*The information-theoretic foundations of LLM training — why minimising NLL equals minimising KL divergence*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Information Theory for LLMs?](#11-why-information-theory-for-llms)
   - [1.2 The Three Key Quantities](#12-the-three-key-quantities)
2. [Entropy H(p)](#2-entropy-hp)
   - [2.1 Definition and Intuition](#21-definition-and-intuition)
   - [2.2 Properties](#22-properties)
   - [2.3 Numerical Examples](#23-numerical-examples)
3. [Cross-Entropy H(p, q)](#3-cross-entropy-hp-q)
   - [3.1 Definition](#31-definition)
   - [3.2 Connection to NLL Loss](#32-connection-to-nll-loss)
4. [KL Divergence D_KL(p ‖ q)](#4-kl-divergence-d_klp--q)
   - [4.1 Definition and Derivation](#41-definition-and-derivation)
   - [4.2 The Decomposition: CE = Entropy + KL](#42-the-decomposition-ce--entropy--kl)
   - [4.3 Why KL ≥ 0 (Gibbs' Inequality)](#43-why-kl--0-gibbs-inequality)
   - [4.4 Asymmetry of KL](#44-asymmetry-of-kl)
5. [Connection to LLM Training](#5-connection-to-llm-training)
   - [5.1 What We Actually Minimise](#51-what-we-actually-minimise)
   - [5.2 The Irreducible Loss](#52-the-irreducible-loss)
6. [Common Mistakes](#6-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Information Theory for LLMs?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM training = making model distribution Q close to true P │
│                                                             │
│  TRUE distribution P: P("mat"|"the cat sat on the") = 0.15 │
│  MODEL distribution Q: Q("mat"|"the cat sat on the") = 0.05│
│                                                             │
│  HOW FAR APART are P and Q?                                 │
│    → KL divergence: D_KL(P ‖ Q) measures this distance!    │
│                                                             │
│  Training MINIMISES: cross-entropy H(P, Q) = H(P) + D_KL(P‖Q)│
│  Since H(P) is fixed: minimising CE = minimising D_KL.     │
│  → Training makes Q as close to P as possible!             │
│                                                             │
│  THE BEST possible loss = H(P) (the entropy of language).   │
│  No model can do better — this is the irreducible loss.     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Three Key Quantities

```
┌──────────────────────────────────────────────────────────────┐
│  1. ENTROPY H(p): inherent uncertainty in distribution p     │
│     H(p) = -Σₓ p(x) log p(x)                               │
│     "How surprised are you on average?"                      │
│                                                              │
│  2. CROSS-ENTROPY H(p, q): cost of using q to encode p      │
│     H(p, q) = -Σₓ p(x) log q(x)                            │
│     "How many bits if you use wrong code q for true dist p?" │
│     This IS the NLL loss!                                    │
│                                                              │
│  3. KL DIVERGENCE D_KL(p ‖ q): distance from q to p         │
│     D_KL(p ‖ q) = Σₓ p(x) log[p(x)/q(x)]                  │
│     "Extra bits wasted by using q instead of optimal p"      │
│     Always ≥ 0. Zero iff p = q.                             │
│                                                              │
│  FUNDAMENTAL RELATIONSHIP:                                   │
│     H(p, q) = H(p) + D_KL(p ‖ q)                            │
│     ↑ CE loss   ↑ irreducible  ↑ what training reduces      │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Entropy H(p)

### 2.1 Definition and Intuition

```
DEFINITION:
  H(p) = -Σₓ p(x) log p(x) = E_p[-log p(x)]

  Measures the AVERAGE surprise (in nats or bits) when sampling from p.
  log = natural log → nats.  log₂ → bits.

INTUITION:
  -log p(x) = "surprise" when observing x.
  If p(x) = 1:   -log(1) = 0   (no surprise, completely expected)
  If p(x) = 0.5: -log(0.5) = 0.69 nats  (moderate surprise)
  If p(x) = 0.01: -log(0.01) = 4.6 nats  (very surprising!)

  H(p) = AVERAGE surprise = expected number of nats to encode a sample.
```

### 2.2 Properties

```
PROPERTIES:
  1. H(p) ≥ 0  (surprise is non-negative)
  2. H(p) = 0  iff p is deterministic (one outcome has P=1)
  3. H(p) is maximised by the UNIFORM distribution
     For |V| outcomes: H_max = log(|V|)
  4. Adding more possible outcomes → increases entropy
  5. H(p) is CONCAVE in p (mixing distributions increases entropy)
```

### 2.3 Numerical Examples

```
EXAMPLE 1: Fair coin p = [0.5, 0.5]
  H = -(0.5×log0.5 + 0.5×log0.5) = -2×0.5×(-0.693) = 0.693 nats = 1 bit

EXAMPLE 2: Biased coin p = [0.9, 0.1]
  H = -(0.9×log0.9 + 0.1×log0.1) = -(0.9×(-0.105) + 0.1×(-2.303))
    = 0.094 + 0.230 = 0.325 nats = 0.47 bits
  (Less uncertain → lower entropy)

EXAMPLE 3: Next-token prediction with 4 candidates
  p = [0.6, 0.2, 0.1, 0.1]
  H = -(0.6×log0.6 + 0.2×log0.2 + 0.1×log0.1 + 0.1×log0.1)
    = -(0.6×(-0.51) + 0.2×(-1.61) + 0.1×(-2.30) + 0.1×(-2.30))
    = 0.306 + 0.322 + 0.230 + 0.230 = 1.088 nats
  PPL = exp(1.088) = 2.97 (effectively choosing from ~3 candidates)
```

---

## 3. Cross-Entropy H(p, q)

### 3.1 Definition

```
CROSS-ENTROPY:
  H(p, q) = -Σₓ p(x) log q(x) = E_p[-log q(x)]

  Measures: average surprise using model q when TRUE distribution is p.
  Always: H(p, q) ≥ H(p, p) = H(p)  (using wrong model → more surprise)
```

### 3.2 Connection to NLL Loss

```
┌──────────────────────────────────────────────────────────────┐
│  THE NLL LOSS IS EXACTLY THE CROSS-ENTROPY:                  │
│                                                              │
│  L_NLL = -(1/T) Σₜ log P_θ(wₜ | w<t)                       │
│                                                              │
│  With p = empirical data distribution (one-hot at true token)│
│  and q = model's predicted distribution P_θ(·|w<t):         │
│                                                              │
│  L_NLL = H(p_data, P_θ)  ✓                                  │
│                                                              │
│  Minimising NLL = minimising cross-entropy with data.        │
│  = minimising H(p_data) + D_KL(p_data ‖ P_θ)                │
│  = minimising D_KL(p_data ‖ P_θ)  (since H(p_data) is fixed)│
│                                                              │
│  → Training makes model as close to true distribution as     │
│    the model architecture allows!                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. KL Divergence D_KL(p ‖ q)

### 4.1 Definition and Derivation

```
KL DIVERGENCE:
  D_KL(p ‖ q) = Σₓ p(x) log[p(x) / q(x)]
              = Σₓ p(x) [log p(x) - log q(x)]
              = E_p[log p(x)] - E_p[log q(x)]
              = -H(p) + H(p, q)

MEASURES: "extra nats needed when using code based on q to 
encode samples from p, beyond the optimal code based on p."
```

### 4.2 The Decomposition: CE = Entropy + KL

```
FUNDAMENTAL DECOMPOSITION:

  H(p, q) = H(p) + D_KL(p ‖ q)
  ↑ what we minimise   ↑ fixed   ↑ what actually decreases
  (NLL loss)          (entropy of  (divergence: how far
                       language)    model is from truth)

  MINIMUM POSSIBLE LOSS = H(p) (achieved when D_KL = 0, i.e., q = p)
  
  For English text: H(p) ≈ 1.0-1.5 bits/character ≈ 1.5-2.0 nats/token
  → PPL ≈ 4-7 is near the entropy of English (theoretical minimum)
  → Current best LLMs (PPL ≈ 5-6) are approaching this limit!
```

### 4.3 Why KL ≥ 0 (Gibbs' Inequality)

```
PROOF (Jensen's inequality):
  D_KL(p ‖ q) = -Σₓ p(x) log[q(x)/p(x)]
              = E_p[-log(q(x)/p(x))]
              ≥ -log(E_p[q(x)/p(x)])    (Jensen: -log is convex)
              = -log(Σₓ p(x) × q(x)/p(x))
              = -log(Σₓ q(x))
              = -log(1) = 0  ✓

  Equality iff q(x)/p(x) is constant for all x where p(x) > 0
  → iff q = p (distributions are identical).
  
  IMPLICATION: H(p,q) ≥ H(p). No model can achieve lower CE than entropy!
```

### 4.4 Asymmetry of KL

```
KL IS NOT SYMMETRIC: D_KL(p ‖ q) ≠ D_KL(q ‖ p) in general!

  D_KL(p ‖ q): "forward KL" — used in LLM training
    Penalises q for assigning LOW probability where p has HIGH probability.
    "Don't miss any probable events"
    → Makes q spread its mass to cover p's support (mode-covering)

  D_KL(q ‖ p): "reverse KL" — used in variational inference
    Penalises q for assigning HIGH probability where p has LOW probability.
    "Don't waste probability on unlikely events"
    → Makes q concentrate on one mode of p (mode-seeking)

  For LLM training: forward KL is correct because we want the model
  to not be surprised by ANY text that occurs in training data.
```

---

## 5. Connection to LLM Training

### 5.1 What We Actually Minimise

```
IN PRACTICE: we minimise empirical cross-entropy on training data.

  L = -(1/T) Σₜ log P_θ(wₜ|w<t)   ← this is what the code computes

  This is an unbiased estimate of H(p_data, P_θ).
  As training proceeds: L decreases toward H(p_data) from above.
  
  If L ≈ H(p_data): model has learned p_data perfectly (overfitting risk!)
  If L >> H(p_data): model still has high D_KL (underfitting).
```

### 5.2 The Irreducible Loss

```
SCALING LAWS (Kaplan et al.):
  L(N) = E + A/N^α
  
  E = irreducible loss ≈ H(p_data) ≈ 1.7 nats
    (entropy of natural language — cannot be reduced even by infinite model)
  
  A/N^α = reducible loss (decreases as model size N grows)
    For LLaMA-3 8B: A/N^α ≈ 0.1 nats
    Total loss: ≈ 1.8 nats → PPL ≈ 6.0
```

---

## 6. Common Mistakes

```
❌ WRONG: Cross-entropy and KL divergence are the same thing
✓ RIGHT:  H(p,q) = H(p) + D_KL(p‖q). They differ by the constant H(p).
          For optimisation they give the same gradient (H(p) has no gradient).
          But numerically CE ≠ KL unless H(p)=0 (which never happens).

❌ WRONG: KL divergence is a "distance metric"
✓ RIGHT:  KL is NOT a metric because it's not symmetric:
          D_KL(p‖q) ≠ D_KL(q‖p). It also doesn't satisfy triangle inequality.
          It IS a "divergence" — measures how one distribution differs from another.

❌ WRONG: Entropy of English is well-defined and precisely known
✓ RIGHT:  Entropy depends on the MODEL of language. Shannon estimated
          ~1.0-1.5 bits/character. Modern estimates: ~0.7-1.0 bits/char.
          We can only provide upper bounds (any model's CE is an upper bound).
          True entropy requires a perfect model we don't have.

❌ WRONG: A model with lower perplexity is always better
✓ RIGHT:  Lower PPL on TRAINING data can mean overfitting.
          PPL on a held-out TEST set is what matters.
          Also, PPL doesn't measure task-specific abilities (reasoning, etc.).
```

---

## 7. Exercises

1. **Entropy Calculation**: Compute H(p) for p=[0.5, 0.3, 0.1, 0.1] in nats and bits. What is the corresponding perplexity?

2. **KL Divergence**: For p=[0.7, 0.2, 0.1] and q=[0.5, 0.3, 0.2], compute D_KL(p‖q) and D_KL(q‖p). Verify they are different. Which is larger and why?

3. **CE Decomposition**: Compute H(p), H(p,q), and D_KL(p‖q) for p=[0.9, 0.1] and q=[0.6, 0.4]. Verify that H(p,q) = H(p) + D_KL(p‖q).

4. **Irreducible Loss**: If English has entropy H≈1.7 nats/token and LLaMA-3 8B achieves loss L=1.82 nats, what is the remaining D_KL? What PPL does this correspond to? How much better could a perfect model be (in PPL)?
