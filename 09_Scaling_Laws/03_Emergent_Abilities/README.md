# Understanding Emergent Abilities in Large Language Models

*Phase transitions, threshold effects, and capabilities that appear only at scale*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What Are Emergent Abilities?](#11-what-are-emergent-abilities)
   - [1.2 The Scaling Paradox](#12-the-scaling-paradox)
2. [Examples of Emergence](#2-examples-of-emergence)
   - [2.1 Chain-of-Thought Reasoning](#21-chain-of-thought-reasoning)
   - [2.2 In-Context Learning](#22-in-context-learning)
   - [2.3 Word-Level Tasks](#23-word-level-tasks)
3. [Mathematical Framework](#3-mathematical-framework)
   - [3.1 Phase Transition Model](#31-phase-transition-model)
   - [3.2 The Metric Matters Hypothesis](#32-the-metric-matters-hypothesis)
   - [3.3 Smooth vs Sharp Transitions](#33-smooth-vs-sharp-transitions)
4. [Debate: Are Emergent Abilities Real?](#4-debate-are-emergent-abilities-real)
   - [4.1 The Schaeffer et al. Critique](#41-the-schaeffer-et-al-critique)
   - [4.2 Task Decomposition Argument](#42-task-decomposition-argument)
5. [Practical Implications](#5-practical-implications)
   - [5.1 When to Expect Emergence](#51-when-to-expect-emergence)
   - [5.2 Predicting Capabilities](#52-predicting-capabilities)
6. [Common Mistakes](#6-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 What Are Emergent Abilities?

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  EMERGENT ABILITY = capability that is:                      │
│    • ABSENT in small models                                 │
│    • PRESENT in large models                                │
│    • NOT predictable by extrapolating small model behaviour │
│                                                             │
│  Performance vs Model Size (log scale):                     │
│                                                             │
│  Accuracy                                                   │
│  100% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●──── (large)       │
│                                       /                     │
│   50% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─/─ ─ ─ ─               │
│                                   /                         │
│   25% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─/─ ─ ─ ─ ─ ─  SHARP jump!  │
│                               /                             │
│    0% ●────●────●────●────●─/─ ─ ─ ─ ─ ─ ─ ─              │
│        └─── random chance ───┘                              │
│       1B   3B   7B   13B  30B  65B  175B                    │
│                  Model size (parameters)                     │
│                                                             │
│  Below some threshold: model is at CHANCE LEVEL             │
│  Above threshold: model suddenly "gets it"                  │
│  The transition is SHARP, not gradual!                      │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Water doesn't "gradually boil" — it's liquid at 99°C and gas at 100°C. Similarly, LLMs are at random-chance performance up to some model size, then abruptly become capable at that task.

### 1.2 The Scaling Paradox

```
PARADOX:
  Loss scales SMOOTHLY as a power law: L(N) = E + A/N^α
  BUT task performance shows SHARP jumps at critical scales.

  How can smooth loss → discontinuous capability?
  
  ANSWER: a task may require multiple sub-skills simultaneously.
  Each sub-skill improves smoothly, but the TASK requires ALL of them.
  P(task correct) = P(skill_1) × P(skill_2) × ... × P(skill_k)
  
  If each skill has sigmoid improvement with scale:
    Product of sigmoids → SHARPER sigmoid!
    With k=5 skills: transition is nearly a step function.
```

---

## 2. Examples of Emergence

### 2.1 Chain-of-Thought Reasoning

```
3-DIGIT ADDITION (Wei et al. 2022):

  Model size:    1B   3B   7B   13B   62B   175B
  Accuracy:      0%   0%   1%   5%    20%   92%
                 └── at chance ──┘     └─ EMERGENCE ─┘

  WHY EMERGENCE: 3-digit addition requires:
    1. Understanding "add these numbers" (language skill)
    2. Aligning digits by place value (spatial reasoning)
    3. Computing each column sum (arithmetic)
    4. Carrying between columns (multi-step state tracking)
    
  Small models can do 1-2 of these but NOT all 4 simultaneously.
  At ~60B: all sub-skills reach sufficient quality → task works!
```

### 2.2 In-Context Learning

```
FEW-SHOT TASK LEARNING (emergent in GPT-3 175B):

  Prompt: "dog → animal, car → vehicle, piano → "
  Expected: "instrument"

  GPT-2 (1.5B):   random output (doesn't understand the pattern)
  GPT-3 (6.7B):   sometimes correct (unreliable)
  GPT-3 (175B):   correct 85%+ of the time (robust ICL!)

  REQUIREMENT for ICL:
    1. Recognise the input-output pattern in examples
    2. Infer the underlying mapping rule
    3. Apply the rule to the new input
    4. Generate the correct output format
    
  Each sub-skill needs a minimum representation capacity.
```

### 2.3 Word-Level Tasks

```
TASKS WITH DOCUMENTED EMERGENCE (BIG-Bench):

| Task | Threshold | Below | Above |
|------|-----------|-------|-------|
| 3-digit addition | ~60B | <5% | >90% |
| Multi-step arithmetic | ~100B | <10% | >70% |
| Logical deduction (5-step) | ~50B | ~25% | >80% |
| Persian QA | ~60B | <10% | >50% |
| Truthful QA | ~100B+ | <30% | >50% |
| International phonetic alphabet | ~60B | ~0% | >50% |
```

---

## 3. Mathematical Framework

### 3.1 Phase Transition Model

```
MODEL FOR EMERGENCE (probabilistic):

  A task requires k independent sub-skills.
  Each sub-skill i has success probability pᵢ(N) that improves with scale:
    pᵢ(N) = σ(aᵢ × (log N - log Nᵢ))   (sigmoid)
    
  where Nᵢ = critical model size for skill i
  and aᵢ = steepness of transition.

  TASK SUCCESS = ALL sub-skills succeed:
    P_task(N) = ∏ᵢ₌₁ᵏ pᵢ(N)

  With k=5 skills, each with threshold around 10B-50B:
    At N=10B: pᵢ ≈ 0.5 each → P_task = 0.5⁵ = 3% (chance!)
    At N=60B: pᵢ ≈ 0.9 each → P_task = 0.9⁵ = 59% (capable!)
    At N=175B: pᵢ ≈ 0.99 each → P_task = 0.99⁵ = 95% (reliable!)

  The PRODUCT of gradual improvements appears SUDDEN.
```

### 3.2 The Metric Matters Hypothesis

```
SCHAEFFER et al. (2023) — A DIFFERENT EXPLANATION:

  Emergence might be an ARTEFACT of the evaluation metric!

  CLAIM: with SMOOTH metrics (e.g., token-level log-likelihood),
  there IS gradual improvement. Only with SHARP metrics 
  (e.g., exact-match accuracy) does it APPEAR discontinuous.

  EXAMPLE (4-digit multiplication):
    Model gets "456 × 789 = 359784" (correct answer: 359784)
    
    EXACT-MATCH metric:
      Model says "359784" → score = 1 (fully correct)
      Model says "359774" → score = 0 (one digit wrong = ZERO!)
    
    TOKEN-LEVEL metric:
      Model says "359774" → 5/6 digits correct = 83% score
      
    With exact-match: sharp transition from 0→1
    With partial credit: gradual improvement is VISIBLE at smaller scales!

┌─────────────────────────────────────────────────────────────┐
│  IMPLICATION: "emergence" may be partially an artefact of   │
│  binary (all-or-nothing) evaluation metrics, not a true     │
│  phase transition in model capabilities.                    │
│                                                             │
│  HOWEVER: some tasks genuinely require a minimum capacity   │
│  threshold that cannot be revealed by partial-credit metrics.│
│  (e.g., you either understand recursion or you don't.)     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Smooth vs Sharp Transitions

```
WHEN IS EMERGENCE "REAL" vs METRIC ARTEFACT?

  REAL (capability threshold):
    - Multi-step reasoning (need ALL steps correct)
    - In-context learning (need to infer the pattern)
    - Code execution (one bug = total failure)
    
  METRIC ARTEFACT:
    - Knowledge recall (gradual improvement visible in log-prob)
    - Translation quality (BLEU improves smoothly)
    - Summarisation (ROUGE improves gradually)

  PRACTICAL IMPACT:
    If you're choosing model size for a specific task:
    - Check if task has documented emergence threshold
    - If yes: you MUST be above that size (no interpolation)
    - If no: smaller model may be cost-effective with fine-tuning
```

---

## 4. Debate: Are Emergent Abilities Real?

### 4.1 The Schaeffer et al. Critique

```
ARGUMENT AGAINST EMERGENCE (2023 paper):

  1. Re-evaluated all BIG-Bench "emergent" tasks
  2. Used linear and smooth metrics instead of exact-match
  3. Found: almost ALL tasks show smooth improvement
  4. Conclusion: "emergence is a mirage"

COUNTER-ARGUMENTS:
  1. Some tasks genuinely have no partial credit 
     (e.g., correctly following a 5-step algorithm)
  2. From a USER perspective, going from 5% → 95% IS emergence
     (the model becomes USABLE at some threshold)
  3. Even if improvement is "smooth", the PRACTICAL impact is nonlinear
     (a 60% → 62% improvement is less meaningful than 10% → 50%)
```

### 4.2 Task Decomposition Argument

```
WHY EMERGENCE MAKES COMPUTATIONAL SENSE:

  A Transformer with L layers and d dimensions has capacity to:
    - Store O(L × d²) "facts" in weights
    - Compose O(L) sequential operations
    - Process O(d) "features" in parallel

  For a task requiring K sequential reasoning steps:
    Need: L ≥ K  (at minimum)
    With noise and imperfect circuits: L ≈ 3K-5K in practice
    
  For 5-step reasoning: need ~15-25 layers minimum
    GPT-2 (12 layers): below threshold → random on 5-step tasks
    GPT-3 (96 layers): above threshold → capable of 5+ step reasoning
```

---

## 5. Practical Implications

### 5.1 When to Expect Emergence

```
RULE OF THUMB (from empirical observations):

  Simple recall/classification: emerges early (~1B)
  Pattern matching / ICL: ~10-30B  
  Multi-step reasoning: ~50-100B
  Complex mathematical proofs: ~100B+ (or not yet)
  Novel scientific insight: not observed (even at largest scales)
```

### 5.2 Predicting Capabilities

```
CANNOT predict emergence from smaller models!
  This is the key practical challenge:

  You train a 7B model → it can't do task X.
  Should you scale to 70B?
  
  ANSWER: you DON'T KNOW if 70B will work either.
  The threshold could be at 13B, 70B, or 200B.
  
  SOLUTIONS:
  1. Check if similar tasks have documented thresholds
  2. Use loss as a PROXY (even if accuracy is flat, is loss decreasing?)
  3. If loss is improving on the task's tokens → larger model WILL work eventually
  4. Fine-tuning can shift the threshold DOWN by 3-5× (a fine-tuned 7B may
     achieve what a base 30B does)
```

---

## 6. Common Mistakes

```
❌ WRONG: Emergent abilities prove LLMs are developing consciousness/AGI
✓ RIGHT:  Emergence just means "non-linear improvement with scale on specific
          benchmarks". It's a statistical phenomenon, not evidence of
          understanding or consciousness. The model still predicts next tokens.

❌ WRONG: If a task is emergent at 60B, you need at least 60B params
✓ RIGHT:  Fine-tuning, chain-of-thought prompting, and better training data
          can shift the emergence threshold DOWN significantly.
          A fine-tuned 7B can often match a base 60B on specific tasks.

❌ WRONG: All improvements in LLMs are emergent (sudden jumps)
✓ RIGHT:  Most improvements are SMOOTH with scale (following power laws).
          Only a minority of tasks (those requiring simultaneous multi-skill
          composition) show sharp transitions. Perplexity always improves smoothly.

❌ WRONG: Once a capability emerges, it's always reliable
✓ RIGHT:  At the emergence threshold, performance is ~50-70% (unreliable).
          Reliable performance (>95%) requires significantly more scale
          beyond the threshold. Emergence is the start, not the finish.
```

---

## 7. Exercises

1. **Product of Sigmoids**: If a task requires 4 independent sub-skills, each with pᵢ(N) = σ(5×(log₁₀N - 10)), compute P_task at N=1B, 10B, 30B, 100B. Plot or describe the shape. At what N does P_task first exceed 50%?

2. **Metric Sensitivity**: A model gets 6-digit multiplication correct 0% of the time (exact match) but gets 4.5/6 digits correct on average. Compute the exact-match score and the "digit accuracy" score. How do these tell a different story about model capability?

3. **Threshold Prediction**: Model A (7B) achieves 5% on task X. Model B (13B) achieves 8%. Model C (30B) achieves 45%. Estimate the emergence threshold. Would you recommend training a 70B model for this task?

4. **Fine-tuning Shift**: A base model shows emergence at 60B. If fine-tuning typically shifts the threshold by 3-5×, what size fine-tuned model might match the 60B base? How many fine-tuning examples might be needed?
