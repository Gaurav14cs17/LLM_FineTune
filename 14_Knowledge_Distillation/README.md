# Understanding Knowledge Distillation for LLMs

*Transferring intelligence from large teacher models to small efficient students*

---

**Knowledge Distillation** trains a small "student" model to mimic the behaviour of a large "teacher" model. Instead of training from scratch on raw data, the student learns from the teacher's rich output distributions — capturing "dark knowledge" that hard labels cannot convey. This is how models like Phi-3, Gemma-2B, and Llama-3.2 1B achieve remarkable quality at small sizes.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Distill?](#11-why-distill)
   - [1.2 Hard Labels vs Soft Labels](#12-hard-labels-vs-soft-labels)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The Distillation Loss](#2-the-distillation-loss)
   - [2.1 KL Divergence Objective](#21-kl-divergence-objective)
   - [2.2 Temperature Scaling — Why and How](#22-temperature-scaling--why-and-how)
   - [2.3 Full Distillation Loss (Combined)](#23-full-distillation-loss-combined)
3. [LLM-Specific Distillation](#3-llm-specific-distillation)
   - [3.1 Sequence-Level Distillation](#31-sequence-level-distillation)
   - [3.2 On-Policy vs Off-Policy](#32-on-policy-vs-off-policy)
4. [Synthetic Data Generation](#4-synthetic-data-generation)
   - [4.1 Teacher as Data Generator](#41-teacher-as-data-generator)
   - [4.2 Quality Filtering](#42-quality-filtering)
5. [Summary](#5-summary)
   - [5.1 Formulas Quick Reference](#51-formulas-quick-reference)
   - [5.2 Common Mistakes](#52-common-mistakes)
6. [Exercises](#6-exercises)

---

## 1. Overview

### 1.1 Why Distill?

```
┌─────────────────────────────────────────────────────────────┐
│  THE DEPLOYMENT PROBLEM:                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Teacher (LLaMA-3 70B):                                     │
│    Quality:     Excellent (MMLU 82%)                        │
│    Latency:     200ms/token                                 │
│    Cost:        $0.001/token                                │
│    Memory:      140 GB (BF16)                               │
│    Hardware:    2× A100 minimum                             │
│                                                             │
│  Student (distilled 1.3B):                                  │
│    Quality:     Good (MMLU 72%) — 88% of teacher!           │
│    Latency:     10ms/token (20× faster!)                    │
│    Cost:        $0.00005/token (20× cheaper!)               │
│    Memory:      2.6 GB (BF16)                               │
│    Hardware:    single RTX 3060                             │
│                                                             │
│  For edge devices (phones, embedded): distilled models      │
│  are the ONLY option. A 70B model cannot run on a phone.   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Hard Labels vs Soft Labels

```
┌─────────────────────────────────────────────────────────────┐
│  HARD LABEL (standard training):                            │
│  "The cat sat on the ___"  → answer: "mat" (one-hot)       │
│  [0, 0, ..., 1, ..., 0]  ← only "mat" has probability 1   │
│                                                             │
│  SOFT LABEL (teacher distribution):                         │
│  "The cat sat on the ___"  → teacher output:                │
│  ┌────────┬──────────────────────────────────────────────┐ │
│  │ "mat"  │███████████████████████████████████  P=0.65   │ │
│  │ "rug"  │████████████                        P=0.15   │ │
│  │ "floor"│█████                               P=0.08   │ │
│  │ "bed"  │███                                 P=0.05   │ │
│  │ "couch"│██                                  P=0.03   │ │
│  │ (rest) │█                                   P=0.04   │ │
│  └────────┴──────────────────────────────────────────────┘ │
│                                                             │
│  THE SOFT LABEL CONTAINS MUCH MORE INFORMATION:             │
│  - "mat" and "rug" are SIMILAR (both floor coverings)       │
│  - "floor" and "bed" are possible but less common            │
│  - "car" and "moon" have ~0 probability (VERY wrong)       │
│                                                             │
│  This inter-class similarity is the "DARK KNOWLEDGE"        │
│  that hard labels cannot express!                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Pipeline Summary

```
DISTILLATION PIPELINE:

  Step 1: Train/obtain TEACHER (large model, high quality)
      ↓
  Step 2: Generate teacher outputs on training corpus
    For each input x: compute teacher distribution P_T(·|x)
      ↓
  Step 3: Train STUDENT to match teacher distribution
    Minimise: KL(P_T(·|x) ‖ P_S(·|x))  per token
      ↓
  Step 4: (Optional) Also train on hard labels
    L_total = α × L_distill + (1-α) × L_NLL
      ↓
  OUTPUT: Small student model with teacher-like behaviour
```

---

## 2. The Distillation Loss

### 2.1 KL Divergence Objective

```
GOAL: make student distribution close to teacher distribution.

  L_KD = KL(P_T ‖ P_S) = Σᵥ P_T(v|x) × log[P_T(v|x) / P_S(v|x)]
                         = Σᵥ P_T(v|x) × [log P_T(v|x) - log P_S(v|x)]
  
  Since P_T is fixed (teacher is frozen):
  Minimising KL(P_T ‖ P_S) w.r.t. student = minimising cross-entropy:
  
  L_KD = -Σᵥ P_T(v|x) × log P_S(v|x)   + const (teacher entropy)
                ↑ teacher prob    ↑ student log-prob
  
  THIS IS JUST CROSS-ENTROPY with SOFT labels P_T instead of one-hot!
```

### 2.2 Temperature Scaling — Why and How

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Teacher is too confident!                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Teacher output (τ=1):  [0.95, 0.03, 0.01, 0.005, ...]    │
│  Almost all information in one token ("mat" = 0.95)         │
│  Other tokens carry negligible gradient signal.             │
│                                                             │
│  WITH TEMPERATURE τ=4: [0.45, 0.18, 0.12, 0.09, ...]      │
│  Now dark knowledge is VISIBLE: "rug" and "floor" get       │
│  meaningful probability → student receives richer signal.   │
└─────────────────────────────────────────────────────────────┘

TEMPERATURE-SCALED SOFTMAX:

  P_T(v|x; τ) = exp(z_v / τ) / Σⱼ exp(z_j / τ)
  
  where z_v = teacher logit for token v
  τ = temperature (τ > 1 softens, τ < 1 sharpens)

EFFECT ON DISTRIBUTION:
  τ → 0:   argmax (hard label) — all weight on top token
  τ = 1:   standard softmax (normal teacher output)
  τ → ∞:   uniform distribution (all tokens equal)
  τ = 2-4: sweet spot for distillation (dark knowledge visible)

GRADIENT SCALING:
  When using temperature τ, gradients are scaled by 1/τ².
  To compensate: multiply the distillation loss by τ²:
  
  L_KD = τ² × CE(P_T(·; τ), P_S(·; τ))
```

### 2.3 Full Distillation Loss (Combined)

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  L_total = α × τ² × CE(P_T(·;τ), P_S(·;τ))                │
│          + (1-α) × CE(y_hard, P_S(·;τ=1))                   │
│            ↑ soft label loss    ↑ hard label loss            │
│                                                              │
│  α ∈ [0, 1]: balance between teacher mimicry and data fit  │
│  τ: temperature for softening teacher distributions         │
│                                                              │
│  TYPICAL VALUES:                                             │
│    α = 0.5 - 0.9 (weight distillation higher)               │
│    τ = 2 - 4 (moderate softening)                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. LLM-Specific Distillation

### 3.1 Sequence-Level Distillation

```
TOKEN-LEVEL (standard):
  At each position t: match P_T(wₜ|w<t) with P_S(wₜ|w<t)
  Both see the GROUND-TRUTH prefix w<t.

SEQUENCE-LEVEL (Kim & Rush 2016):
  Step 1: Generate full sequence from teacher: ŷ ~ P_T(·|x)
  Step 2: Train student on (x, ŷ) pairs with standard NLL

  ADVANTAGE: student learns from TEACHER-GENERATED text,
  not just teacher probabilities at each position.
  Simpler (no KL computation, just SFT on teacher outputs).
  
  DISADVANTAGE: loses the full distribution information.
  Only captures the MODE of the teacher (argmax behaviour).

ON-POLICY DISTILLATION (best of both):
  Step 1: Student generates: ŷ_S ~ P_S(·|x)
  Step 2: Score with teacher: P_T(ŷ_S|x)
  Step 3: Update student to increase P_S(ŷ_S|x) weighted by P_T(ŷ_S|x)
  
  This trains the student on ITS OWN distribution,
  avoiding the "exposure bias" problem.
```

### 3.2 On-Policy vs Off-Policy

```
OFF-POLICY (simpler, more common):
  Train student on data generated BEFORE training.
  Data source: teacher outputs on a fixed dataset.
  
  + Simple: just SFT on teacher-generated data
  + Efficient: generate once, train many times
  - Distribution mismatch: student sees teacher's style, not its own errors

ON-POLICY (harder, better quality):
  At each step:
    1. Student generates response to prompt
    2. Teacher scores/corrects the student's response
    3. Student learns from the feedback
  
  + No distribution mismatch (student learns from its own mistakes)
  + Better for long-sequence generation
  - Expensive: requires teacher inference at every training step
  - Harder to implement and parallelize
```

---

## 4. Synthetic Data Generation

### 4.1 Teacher as Data Generator

```
MODERN DISTILLATION (Phi-3, Orca, WizardLM):
  Instead of matching distributions, generate HIGH-QUALITY data from teacher.

  Step 1: Create diverse prompts (seed instructions)
  Step 2: Teacher generates detailed responses
  Step 3: Filter for quality (see below)
  Step 4: Train student via standard SFT on filtered data

  SCALE: typically 1M-10M teacher-generated examples.
  
  DATA DIVERSITY strategies:
  - Evol-Instruct (WizardLM): evolve prompts to be harder/more diverse
  - Self-Instruct: teacher generates its own prompts
  - Topic-guided: generate across different domains systematically
  - Difficulty-scaled: easy → medium → hard progression
```

### 4.2 Quality Filtering

```
NOT ALL TEACHER OUTPUTS ARE GOOD — filter aggressively:

  Filter 1: SELF-CONSISTENCY
    Generate N responses to same prompt.
    Keep only prompts where ≥ 80% of responses agree.
    (Disagreement = teacher is uncertain = bad training signal)
  
  Filter 2: REWARD MODEL SCORING
    Score each (prompt, response) with a reward model.
    Keep only top-P% (typically P=50-70).
  
  Filter 3: LENGTH FILTER
    Remove very short responses (< 50 tokens): likely low quality.
    Remove very long responses (> 2000 tokens): likely rambling.
  
  Filter 4: DECONTAMINATION
    Remove examples that overlap with evaluation benchmarks.
    (Prevents data leakage → inflated scores)
  
  TYPICAL PIPELINE:
    10M raw teacher outputs → 3M after filtering → train student
```

---

## 5. Summary

### 5.1 Formulas Quick Reference

**Standard distillation loss:**

```
L_KD = τ² × CE(softmax(z_T/τ), softmax(z_S/τ))
```

**Combined loss:**

```
L = α × L_KD + (1-α) × L_NLL(y_hard, P_S)
```

**Temperature effect:**

```
P(v; τ) = exp(z_v/τ) / Σⱼ exp(z_j/τ)
  τ=1: standard softmax
  τ>1: softer (more uniform)
  τ<1: sharper (more peaked)
```

| Method | Data needed | Quality | Use case |
|--------|-------------|---------|----------|
| Token-level KD | Teacher logits | Best | White-box teacher |
| Sequence-level KD | Teacher samples | Good | Black-box teacher |
| Synthetic data SFT | Teacher responses | Good | API-only teacher |
| On-policy KD | Live teacher scoring | Best | Resource-rich |

### 5.2 Common Mistakes

```
❌ WRONG: A 1B student can match a 70B teacher with enough distillation
✓ RIGHT:  The student has a CAPACITY CEILING. A 1B model cannot represent
          all the knowledge of 70B. Expect 85-95% of teacher quality,
          not 100%. The ceiling depends on student architecture and size.

❌ WRONG: Temperature τ=1 is fine for distillation
✓ RIGHT:  τ=1 makes the teacher too peaked (95% on one token).
          The student gets almost no gradient from non-top tokens.
          τ=2-4 reveals dark knowledge → much better learning signal.

❌ WRONG: More synthetic data is always better
✓ RIGHT:  Quality > quantity for distillation data.
          1M high-quality filtered examples often beats 10M unfiltered.
          Garbage-in → garbage-out applies strongly to distillation.

❌ WRONG: Distillation replaces pretraining
✓ RIGHT:  Best results come from pretrained student + distillation fine-tuning.
          Pretraining gives the student general language understanding.
          Distillation fine-tuning transfers specific teacher capabilities.
```

---

## 6. Exercises

1. **Temperature Effect**: For teacher logits z=[5.0, 2.0, 1.0, 0.5], compute P(v) at τ=1, τ=2, and τ=4. How does the entropy change? At which τ is the "dark knowledge" most visible?

2. **Distillation Loss**: For τ=2, teacher logits z_T=[3.0, 1.0, 0.5] and student logits z_S=[2.5, 1.2, 0.8]: compute L_KD = τ² × CE(P_T(·;τ), P_S(·;τ)). Show the soft distributions for both.

3. **Capacity Analysis**: A 1B student trained on N examples from a 70B teacher. Estimate the minimum N needed if the student needs ~5× more data per parameter than pretraining to capture teacher knowledge. (Hint: Chinchilla says ~20 tokens/param for pretraining.)

4. **Quality Filtering**: You generate 5M synthetic examples from a teacher. Self-consistency filter keeps 70%, reward model keeps 60% of those, length filter keeps 90% of those. How many examples remain? Is this enough to fine-tune a 3B student?
