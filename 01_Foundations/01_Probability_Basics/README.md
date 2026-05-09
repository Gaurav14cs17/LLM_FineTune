# Understanding Probability Basics for LLMs

*The mathematical backbone of every language model*

---

Probability theory is the foundation of every language model. Without understanding how to measure uncertainty, chain events, and update beliefs, none of the training objectives, sampling methods, or evaluation metrics of modern LLMs make sense.

This guide walks through probability from first principles up to the formulas used in GPT and LLaMA training.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Probability?](#11-why-probability)
   - [1.2 Key Concepts at a Glance](#12-key-concepts-at-a-glance)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Sample Spaces and Events](#2-sample-spaces-and-events)
   - [2.1 Formal Definitions](#21-formal-definitions)
   - [2.2 Discrete PMF](#22-discrete-pmf)
3. [Conditional Probability](#3-conditional-probability)
   - [3.1 Definition and Intuition](#31-definition-and-intuition)
   - [3.2 Product Rule](#32-product-rule)
   - [3.3 Bayes' Theorem](#33-bayes-theorem)
4. [The Chain Rule](#4-the-chain-rule)
   - [4.1 Statement and Proof](#41-statement-and-proof)
   - [4.2 LLM Connection](#42-llm-connection)
5. [Expectation and Variance](#5-expectation-and-variance)
   - [5.1 Expectation](#51-expectation)
   - [5.2 Variance and the Attention Scaling Proof](#52-variance-and-the-attention-scaling-proof)
6. [Summary](#6-summary)
   - [6.1 All Formulas Quick Reference](#61-all-formulas-quick-reference)
   - [6.2 Common Mistakes](#62-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Probability?

Every token an LLM generates is a **sample from a probability distribution**. The model's output is not a single number — it's a full distribution over the 128,000-token vocabulary.

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION: LLM as a Probability Machine                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "The Eiffel Tower is located in"                    │
│                                                             │
│  Output distribution over all tokens:                      │
│   "Paris"    │██████████████████████████│  P = 0.72        │
│   "France"   │████████                  │  P = 0.12        │
│   "the"      │████                      │  P = 0.06        │
│   "downtown" │██                        │  P = 0.03        │
│   (others)   │█                         │  P = 0.07        │
│                                                             │
│  The model SAMPLES from this distribution (or takes argmax) │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: A weather forecast doesn't say "it WILL rain." It says "70% chance of rain." An LLM is similar — it assigns probabilities to all possible next words, and we choose one.

### 1.2 Key Concepts at a Glance

| Concept | Symbol | What It Means | LLM Use |
|---------|--------|---------------|---------|
| Probability | P(A) | How likely event A is | P(next token = "Paris") |
| Joint probability | P(A, B) | Both A and B occur | P(w₁, w₂, ..., wₙ) |
| Conditional | P(A \| B) | A given B already happened | P(wₜ \| w₁,...,wₜ₋₁) |
| Expectation | E[X] | Average value of X | Expected loss per step |
| Variance | Var[X] | Spread of X | Attention score variance |

### 1.3 Pipeline Summary

```
PROBABILITY CONCEPTS USED IN LLMs

P(A) basics
    ↓
P(A|B) conditional probability
    ↓
Bayes' Theorem  ← used in RLHF reward modelling
    ↓
Chain Rule  ← the CORE formula of autoregressive LMs
    ↓
Expectation & Variance  ← used in attention scaling, loss analysis
    ↓
NLL Loss = −E[log P]  ← the training objective
```

---

## 2. Sample Spaces and Events

### 2.1 Formal Definitions

**The Setup:**

```
SAMPLE SPACE Ω = the set of all possible outcomes

For a language model predicting the next token:
  Ω = V = {0, 1, 2, ..., |V|−1}   (all vocabulary token IDs)

EVENT A ⊆ Ω = any subset of outcomes
  Example: A = {"Paris", "London", "Berlin"}  (the event "a capital city is next")

PROBABILITY MEASURE P: assigns a number in [0, 1] to each event

Three axioms (Kolmogorov 1933):
  (i)   P(Ω) = 1                         always certain something happens
  (ii)  P(A) ≥ 0                          probabilities are non-negative
  (iii) P(A ∪ B) = P(A) + P(B)           if A ∩ B = ∅  (disjoint addition)
```

### 2.2 Discrete PMF

A **Probability Mass Function** assigns a probability to each discrete outcome:

```
PMF conditions:
  P(X = x) ≥ 0       for all x ∈ Ω
  Σ P(X = x) = 1     sum over all outcomes = 1
```

**Example — Token Distribution:**

```
Vocabulary: {"cat", "dog", "bird", "fish", "ant"}

┌────────┬───────────────────────────────────────────┬────────┐
│ Token  │ Probability Bar                           │  P(x)  │
├────────┼───────────────────────────────────────────┼────────┤
│ "cat"  │ ████████████████████████████              │  0.35  │
│ "dog"  │ ████████████████████                      │  0.25  │
│ "bird" │ ████████████                              │  0.15  │
│ "fish" │ ████████████                              │  0.15  │
│ "ant"  │ ████████                                  │  0.10  │
└────────┴───────────────────────────────────────────┴────────┘
                                              Sum =    1.00 ✓
```

**Variables:**

| Symbol | Meaning |
|--------|---------|
| Ω | Sample space (all possible outcomes) |
| A, B | Events (subsets of Ω) |
| P(X = x) | Probability that random variable X equals x |
| V | Vocabulary (sample space for token prediction) |

---

## 3. Conditional Probability

### 3.1 Definition and Intuition

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION: Shrinking the Sample Space                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEFORE conditioning:                                       │
│  Full universe Ω  =  {"cat","dog","fish","car","truck"}     │
│  P("cat") = 0.20                                            │
│                                                             │
│  AFTER conditioning on "The pet is a ___":                  │
│  Relevant universe = {"cat","dog","fish"}   (animals only)  │
│  P("cat" | "The pet is a ___") = 0.40   ← higher!           │
│                                                             │
│  Conditioning RESTRICTS the sample space, then renormalises │
└─────────────────────────────────────────────────────────────┘
```

**Formal Definition:**

```
P(A | B) = P(A ∩ B) / P(B)       (for P(B) > 0)
           ─────────────
           "joint probability divided by the probability of the condition"
```

**Numerical Example:**

```
P("cat", "sat") = 0.03   (both appear in sequence)
P("sat")        = 0.05   (marginal probability)

P("cat" | "sat") = 0.03 / 0.05 = 0.60
→ "Given 'sat' appeared, the word before it is 'cat' with 60% probability"
```

### 3.2 Product Rule

Rearranging the conditional probability definition:

```
P(A, B) = P(A | B) × P(B)
         = P(B | A) × P(A)

Both are valid — they're just different orderings of the same joint probability.
```

### 3.3 Bayes' Theorem

**Derivation from scratch:**

```
Step 1:  P(A | B) = P(A, B) / P(B)            (definition)
Step 2:  P(B | A) = P(A, B) / P(A)            (definition)
Step 3:  From Step 2: P(A, B) = P(B | A) × P(A)
Step 4:  Substitute into Step 1:

┌──────────────────────────────────────────────────────────┐
│          P(B | A) × P(A)                                 │
│ P(A|B) = ─────────────────                               │
│               P(B)                                       │
└──────────────────────────────────────────────────────────┘

where P(B) = Σₐ P(B | A=a) × P(A=a)   (total probability theorem)
```

**The Four Components:**

| Term | Name | Meaning |
|------|------|---------|
| P(A) | Prior | Belief about A *before* seeing B |
| P(B \| A) | Likelihood | How probable is evidence B if A is true? |
| P(B) | Evidence | Overall probability of seeing B (normaliser) |
| P(A \| B) | Posterior | Updated belief after seeing B |

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION: Information Flow                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prior belief    Evidence arrives    Updated belief        │
│  P(A) = 0.20   ─────────────────►   P(A|B) = 0.60         │
│                      B                                      │
│                 "new information     "posterior"            │
│                  updates belief"                            │
│                                                             │
│  LLM application:                                          │
│  P(meaning | text) ∝ P(text | meaning) × P(meaning)        │
│  ↑ semantic inference  ↑ language model    ↑ prior          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. The Chain Rule

### 4.1 Statement and Proof

**Theorem:** For any sequence of random variables w₁, w₂, ..., wₙ:

```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)

                    = ∏ₜ₌₁ⁿ P(wₜ | w₁, ..., wₜ₋₁)
```

**Proof by induction:**

```
BASE CASE (n=2):
  P(w₁, w₂) = P(w₂ | w₁) × P(w₁)    ← directly from product rule ✓

INDUCTIVE STEP:
  Assume: P(w₁,...,wₙ₋₁) = ∏ₜ₌₁ⁿ⁻¹ P(wₜ | w₁,...,wₜ₋₁)

  Then:
  P(w₁,...,wₙ) = P(wₙ | w₁,...,wₙ₋₁) × P(w₁,...,wₙ₋₁)    ← product rule
               = P(wₙ | w₁,...,wₙ₋₁) × ∏ₜ₌₁ⁿ⁻¹ P(wₜ|w₁,...,wₜ₋₁)   ← induction
               = ∏ₜ₌₁ⁿ P(wₜ | w₁,...,wₜ₋₁)    QED ■
```

### 4.2 LLM Connection

```
┌─────────────────────────────────────────────────────────────┐
│  TIMELINE: How LLM Generation Uses the Chain Rule           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  t=1       t=2           t=3              t=4              │
│  ┌───┐  ┌───────┐  ┌────────────┐  ┌─────────────────┐    │
│  │ w₁│─►│w₁  w₂│─►│w₁  w₂  w₃ │─►│w₁  w₂  w₃  w₄  │    │
│  └───┘  └───────┘  └────────────┘  └─────────────────┘    │
│     ↓         ↓            ↓                 ↓             │
│  P(w₁)  P(w₂|w₁)  P(w₃|w₁,w₂)    P(w₄|w₁,w₂,w₃)        │
│                                                             │
│  P(w₁,w₂,w₃,w₄) = P(w₁) × P(w₂|w₁) × P(w₃|w₁w₂) × ...  │
│                                                             │
│  This IS autoregressive generation — one token at a time!  │
└─────────────────────────────────────────────────────────────┘
```

**Why it matters for training:**

```
TRAINING LOSS uses the chain rule directly:

  L(θ) = -1/T × Σₜ log P_θ(wₜ | w₁,...,wₜ₋₁)
         ↑
         average negative log probability of each token given its context

Step 1: Compute P_θ(wₜ | context) via transformer forward pass
Step 2: Take log → negative log = "surprise" at this token
Step 3: Average over all T tokens in the sequence
Step 4: Minimise this loss → model gets better at predicting each next token
```

---

## 5. Expectation and Variance

### 5.1 Expectation

**Definition:**

```
E[X] = Σₓ  x × P(X = x)     (discrete)
     = ∫ x × p(x) dx         (continuous)
```

**Linearity of Expectation** (works even for *dependent* variables):

```
E[aX + bY + c] = a×E[X] + b×E[Y] + c

PROOF:
  E[aX + bY] = Σₓᵧ (ax + by) P(X=x, Y=y)
             = a Σₓ x [Σᵧ P(X=x, Y=y)] + b Σᵧ y [Σₓ P(X=x, Y=y)]
             = a Σₓ x P(X=x) + b Σᵧ y P(Y=y)
             = a E[X] + b E[Y]   ■
```

### 5.2 Variance and the Attention Scaling Proof

**Definition:**

```
Var[X] = E[(X − E[X])²] = E[X²] − (E[X])²
```

**The critical proof — why attention uses 1/√d_k:**

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: The dot product q·k grows with dimension d_k!     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  s = q·k = Σᵢ₌₁^{d_k} qᵢ × kᵢ                             │
│                                                             │
│  If qᵢ ~ N(0,1) and kᵢ ~ N(0,1) independently:            │
│                                                             │
│  E[s] = Σᵢ E[qᵢ kᵢ] = Σᵢ E[qᵢ]E[kᵢ] = 0                  │
│                                                             │
│  Var[s] = Σᵢ Var[qᵢ kᵢ]                                   │
│         = Σᵢ (E[qᵢ²]E[kᵢ²] - (E[qᵢkᵢ])²)                 │
│         = Σᵢ (1×1 - 0)                                     │
│         = d_k                                               │
│                                                             │
│  std[s] = √d_k   ← GROWS with dimension!                   │
│                                                             │
│  For d_k=64: std=8 → scores like [8, 3, -4, ...]            │
│  Softmax of [8,3,-4] ≈ [0.997, 0.003, 0.0] → almost one-hot│
│  → Gradient of softmax ≈ 0 → VANISHING GRADIENTS!          │
│                                                             │
│  FIX: divide by √d_k → Var[s/√d_k] = Var[s]/d_k = 1 ✓    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Summary

### 6.1 All Formulas Quick Reference

**Conditional Probability:**

```
P(A | B) = P(A ∩ B) / P(B)
```

**Bayes' Theorem:**

```
P(A | B) = P(B | A) × P(A) / P(B)
```

**Chain Rule:**

```
P(w₁,...,wₙ) = ∏ₜ₌₁ⁿ P(wₜ | w₁,...,wₜ₋₁)
```

**Expectation:**

```
E[X] = Σₓ x × P(X=x)
E[aX + bY] = a E[X] + b E[Y]   (linearity)
```

**Variance:**

```
Var[X] = E[X²] − (E[X])²
Var[Σᵢ Xᵢ] = Σᵢ Var[Xᵢ]     if independent
```

**NLL Training Loss:**

```
L(θ) = −(1/T) Σₜ log P_θ(wₜ | w₁,...,wₜ₋₁)
```

| Concept | LLM Application |
|---------|----------------|
| Chain rule | Autoregressive generation formula |
| Conditional P | P(next token \| all previous tokens) |
| NLL loss | Every transformer training loop |
| Var[q·k] = d_k | Why we scale attention by 1/√d_k |
| Bayes | RLHF reward modelling, posterior inference |

### 6.2 Common Mistakes

```
❌ WRONG: P(A|B) = P(B|A)   (confusing posterior and likelihood)
✓ RIGHT:  P(A|B) = P(B|A) × P(A) / P(B)   (Bayes' theorem)

❌ WRONG: Var[X+Y] = Var[X] + Var[Y]  (always)
✓ RIGHT:  Var[X+Y] = Var[X] + Var[Y] + 2Cov[X,Y]  (in general)
          Only true when X and Y are independent.

❌ WRONG: The chain rule only applies to two variables
✓ RIGHT:  It applies to any number of variables: P(w₁,...,wₙ) = ∏ P(wₜ|w<t)

❌ WRONG: Higher probability = better token choice (always)
✓ RIGHT:  Temperature sampling can improve generation diversity
          even at the cost of per-step likelihood.
```

---

## 7. Exercises

1. **Chain Rule Proof**: Work through the full inductive proof for n=3 explicitly. Write out P(w₁,w₂,w₃) as a product of three conditional probabilities.

2. **Bigram Calculation**: A bigram model has P("cat"|"the") = 0.05 and P("the") = 0.08. Compute P("the","cat") using the product rule.

3. **Attention Scaling**: For d_k = 128, compute Var[q·k] without scaling. What is the standard deviation? Why does this cause problems for softmax?

4. **Uniform LM Perplexity**: A uniform model assigns P(wₜ|context) = 1/|V|. Show that the training loss L = log|V|. For |V|=128000, what is L and PPL?
