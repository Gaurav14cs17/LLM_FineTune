# Understanding Language Models: Definition and Loss Function

*The core identity P(w₁,...,wₙ) = ∏ P(wₜ|w<t) — from probability to training objective*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What IS a Language Model?](#11-what-is-a-language-model)
   - [1.2 Autoregressive Factorisation](#12-autoregressive-factorisation)
2. [The Chain Rule of Probability](#2-the-chain-rule-of-probability)
   - [2.1 Derivation](#21-derivation)
   - [2.2 Why Autoregressive (Left-to-Right)?](#22-why-autoregressive-left-to-right)
3. [Negative Log-Likelihood (NLL) Loss](#3-negative-log-likelihood-nll-loss)
   - [3.1 Maximum Likelihood Estimation](#31-maximum-likelihood-estimation)
   - [3.2 NLL as the Training Objective](#32-nll-as-the-training-objective)
   - [3.3 Numerical Example](#33-numerical-example)
4. [Perplexity](#4-perplexity)
   - [4.1 Definition and Intuition](#41-definition-and-intuition)
   - [4.2 Relationship to Cross-Entropy](#42-relationship-to-cross-entropy)
   - [4.3 Interpreting PPL Values](#43-interpreting-ppl-values)
5. [Common Mistakes](#5-common-mistakes)
6. [Exercises](#6-exercises)

---

## 1. Overview

### 1.1 What IS a Language Model?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  A Language Model assigns a PROBABILITY to any sequence     │
│  of tokens. Higher probability = more "natural" text.       │
│                                                             │
│  P("The cat sat on the mat") = 0.0001   (high — natural)   │
│  P("Cat the on sat mat the") = 0.000000001 (low — garbage) │
│                                                             │
│  EQUIVALENTLY, it predicts the next token given a prefix:   │
│                                                             │
│  P(next_token | "The cat sat on the") =                     │
│    "mat":   0.15                                            │
│    "floor": 0.08                                            │
│    "bed":   0.05                                            │
│    "roof":  0.02                                            │
│    ... (128,256 tokens in vocabulary, all sum to 1.0)       │
│                                                             │
│  AN LLM IS: a neural network that estimates this P(·|·).   │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: A language model is like autocomplete on your phone. Given "I'm going to the...", it assigns probabilities to all possible next words. The model IS that probability distribution.

### 1.2 Autoregressive Factorisation

```
FULL SEQUENCE PROBABILITY (the LLM identity):

  P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)
                       = ∏ₜ₌₁ⁿ P(wₜ | w₁, ..., wₜ₋₁)
                       = ∏ₜ₌₁ⁿ P(wₜ | w<t)

  This is EXACT (no approximation!) — it's the chain rule of probability.
  Every autoregressive LLM models exactly this factorisation.
```

---

## 2. The Chain Rule of Probability

### 2.1 Derivation

```
CHAIN RULE (from basic probability axioms):

  P(A, B) = P(A) × P(B|A)                    (definition of conditional)
  P(A, B, C) = P(A) × P(B|A) × P(C|A,B)     (apply twice)
  P(w₁,...,wₙ) = ∏ₜ P(wₜ | w<t)             (apply n-1 times)

  This holds for ANY joint distribution — no assumptions needed.
  The LLM's job: learn a neural network that approximates each P(wₜ|w<t).

EXAMPLE:
  P("The cat sat") = P("The") × P("cat"|"The") × P("sat"|"The cat")
                   ≈ 0.05   ×    0.02          ×     0.03
                   = 0.00003

  log P("The cat sat") = log(0.05) + log(0.02) + log(0.03)
                        = -3.0 + -3.9 + -3.5 = -10.4
```

### 2.2 Why Autoregressive (Left-to-Right)?

```
ALTERNATIVES and why autoregressive wins for generation:

  Masked LM (BERT): P(w_t | w_1,...,w_{t-1}, w_{t+1},...,w_n)
    Can see BOTH past and future! Good for understanding.
    BUT: cannot generate text left-to-right (needs future context).
    
  Autoregressive (GPT): P(w_t | w_1,...,w_{t-1})
    Only sees the past. Can generate token by token!
    Training: teacher forcing (feed ground truth prefix, predict next).
    Inference: sample one token, append, repeat.

  WHY LEFT-TO-RIGHT (not right-to-left)?
    Language is read/written left-to-right (in English).
    Either direction is mathematically valid (chain rule is symmetric).
    XLNet showed right-to-left and permutation orderings work too.
    Convention: left-to-right is simplest and fastest for generation.
```

---

## 3. Negative Log-Likelihood (NLL) Loss

### 3.1 Maximum Likelihood Estimation

```
GOAL: find parameters θ that maximise P_θ(training data).

  θ* = argmax_θ  P_θ(D)
     = argmax_θ  ∏_{(x,y) ∈ D} P_θ(y|x)
     = argmax_θ  Σ_{(x,y) ∈ D} log P_θ(y|x)    (log is monotone)
     = argmin_θ  -Σ_{(x,y) ∈ D} log P_θ(y|x)   (flip sign to minimise)
     = argmin_θ  NLL(θ)
```

### 3.2 NLL as the Training Objective

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  NLL LOSS (the LLM training objective):                      │
│                                                              │
│  L(θ) = -(1/T) × Σₜ₌₁ᵀ log P_θ(wₜ | w<t)                  │
│                                                              │
│  where T = total number of tokens in the batch              │
│  and P_θ(wₜ|w<t) = softmax(logits)[wₜ]                      │
│                                                              │
│  GRADIENT:                                                   │
│  ∂L/∂θ = -(1/T) × Σₜ (1/P_θ(wₜ|w<t)) × ∂P_θ(wₜ|w<t)/∂θ   │
│                                                              │
│  In practice: cross-entropy between one-hot(wₜ) and P_θ(·|w<t)│
│  L(θ) = -(1/T) × Σₜ Σᵥ one_hot(wₜ)[v] × log P_θ(v|w<t)   │
│        = -(1/T) × Σₜ log P_θ(wₜ|w<t)   (since one_hot selects one term)│
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Numerical Example

```
EXAMPLE: vocabulary = {"the":0, "cat":1, "sat":2, "on":3}

  Sequence: "the cat sat on" (4 tokens)
  
  At position 1 (predicting "cat" given "the"):
    Model logits: [1.2, 2.5, 0.3, -0.5]
    Softmax:      [0.14, 0.51, 0.06, 0.03] (sums to ~0.74 + normalisation)
    P("cat"|"the") = 0.51
    -log P = -log(0.51) = 0.67
  
  At position 2 (predicting "sat" given "the cat"):
    P("sat"|"the cat") = 0.35
    -log P = -log(0.35) = 1.05
  
  At position 3 (predicting "on" given "the cat sat"):
    P("on"|"the cat sat") = 0.45
    -log P = -log(0.45) = 0.80

  NLL = (0.67 + 1.05 + 0.80) / 3 = 0.84

  INTERPRETATION: on average, the model assigns exp(-0.84) = 0.43 
  probability to the correct next token. Lower NLL = better model.
```

---

## 4. Perplexity

### 4.1 Definition and Intuition

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  PERPLEXITY (PPL) = exp(NLL) = exp(-(1/T) Σₜ log P(wₜ|w<t))│
│                                                              │
│  INTUITION: "how many tokens is the model choosing between  │
│  at each step, on average?"                                  │
│                                                              │
│  PPL = 1:      model is PERFECT (always P=1 for correct)    │
│  PPL = 10:     like choosing uniformly from 10 options       │
│  PPL = 100:    like choosing from 100 options (uncertain)    │
│  PPL = |V|:    uniform over vocabulary (knows nothing)       │
│                                                              │
│  For LLaMA-3 8B on standard benchmarks: PPL ≈ 6-8           │
│  → The model narrows down the next token to ~7 candidates   │
│    on average (out of 128,256 in the vocabulary!)            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Relationship to Cross-Entropy

```
MATHEMATICAL CONNECTION:

  Cross-entropy: H(p, q) = -Σₓ p(x) log q(x)
  NLL = empirical cross-entropy = H(p_data, P_θ)
  PPL = exp(H(p_data, P_θ)) = 2^{H in bits}

  If we use bits (log₂): PPL = 2^{cross-entropy in bits}
  If we use nats (ln):   PPL = e^{cross-entropy in nats}
  
  Standard: use natural log → PPL = exp(NLL in nats).
```

### 4.3 Interpreting PPL Values

```
TYPICAL PERPLEXITIES:

| Model | PPL (WikiText) | What it means |
|-------|---------------|---------------|
| Random (|V|=128K) | 128,256 | Knows nothing |
| N-gram (n=5) | ~80 | Captures local statistics |
| LSTM (1B params) | ~25 | Understands sentences |
| GPT-2 (1.5B) | ~18 | Good language model |
| LLaMA-2 7B | ~7.5 | Strong language model |
| LLaMA-3 8B | ~6.2 | State-of-the-art for size |
| GPT-4 (~1.8T?) | ~4-5 | Near human-level fluency |

IMPORTANT: PPL is dataset-dependent! 
  PPL on code ≠ PPL on prose ≠ PPL on dialogue.
  Always compare on the SAME evaluation set.
```

---

## 5. Common Mistakes

```
❌ WRONG: Lower perplexity always means better downstream performance
✓ RIGHT:  PPL measures next-token prediction quality (average over ALL tokens).
          A model good at predicting "the" and "a" (frequent) can have low PPL
          while being bad at harder tasks (reasoning, math, coding).
          PPL is necessary but not sufficient for task quality.

❌ WRONG: Training loss = NLL = cross-entropy are all different things
✓ RIGHT:  For autoregressive LLMs, they are ALL THE SAME:
          Training loss = NLL = cross-entropy(one_hot, model_distribution)
          These are just different names for -(1/T) Σ log P(wₜ|w<t).

❌ WRONG: The model is trained to predict one specific next token
✓ RIGHT:  The model is trained to output a DISTRIBUTION over all tokens.
          The loss pushes it to assign HIGH probability to the correct token.
          Many tokens may be valid continuations ("I went to the [store/park/...]").

❌ WRONG: PPL of 6 means the model always has 6 candidates
✓ RIGHT:  PPL is a GEOMETRIC MEAN. For some tokens P≈0.99 (very predictable:
          "New" → "York"), for others P≈0.01 (creative positions: "The story
          was about a..."). PPL=6 is the average across all positions.
```

---

## 6. Exercises

1. **NLL Computation**: For the sequence "I love cats" with P("I")=0.1, P("love"|"I")=0.05, P("cats"|"I love")=0.02: compute NLL (in nats) and perplexity.

2. **Perplexity Interpretation**: Model A has PPL=8 on news articles, PPL=25 on code. Model B has PPL=10 on news, PPL=12 on code. Which model is "better"? Why is this question ill-defined?

3. **Loss Landscape**: If a model assigns P=1/|V|=1/128256 to every token uniformly, what is its PPL? How does this compare to a model that has memorised the training data perfectly (P=1 for correct token)?

4. **Chain Rule Verification**: For a 3-token sequence "A B C", verify that P(A,B,C) = P(A)P(B|A)P(C|A,B) using a concrete example. What happens to log P(A,B,C) as sequence length grows?
