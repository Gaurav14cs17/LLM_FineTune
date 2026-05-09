# 03 — Emergent Capabilities

## 1. What "Emergence" Looks Like

```
SMOOTH scaling (most tasks):
  Accuracy
  ↑
  1.0│                    · · ·
     │              · · ·
  0.5│         · · ·
     │    · · ·
  0.0│· ·
     └─────────────────────────── log(compute / parameters)
     Small models gradually improve — predictable from extrapolation

EMERGENT capability (some tasks):
  Accuracy
  ↑
  1.0│                         · · · ·
     │                       ·
  0.5│                      ·
     │                     ·
  0.0│· · · · · · · · · · ·
     └─────────────────────────── log(compute / parameters)
                            ↑
                    EMERGENCE THRESHOLD
                    Near-zero below, sudden jump above
```

## 2. Real Emergence Examples

```
MULTI-DIGIT ADDITION (3-digit numbers):

  Model size:   ~1B    ~5B   ~10B   ~50B   ~100B
  Accuracy:      0%     0%    0%    68%     97%
  
  ┌──────────────────────────────────────────────────────┐
  │ "What is 347 + 582?"                                 │
  │                                                      │
  │  < 50B params: cannot reliably add 3-digit numbers   │
  │  > 50B params: suddenly solves it with 68%+ accuracy │
  └──────────────────────────────────────────────────────┘

CHAIN-OF-THOUGHT (step-by-step reasoning):

  Model size:   <100B                   >100B
  GSM8K:          ~0%                    ~58%
  
  ┌──────────────────────────────────────────────────────┐
  │ CoT prompting ("Let's think step by step..."):        │
  │   Small models:  CoT HURTS performance               │
  │   Large models:  CoT massively HELPS performance     │
  └──────────────────────────────────────────────────────┘

WHY? Large models can generate coherent reasoning chains.
Small models generate nonsensical steps that mislead the final answer.
```

## 3. The Metric Artifact Theory

```
CLAIM (Schaeffer et al. 2023): Emergence may be an artifact of non-linear metrics.

EXAMPLE: Task requires k=5 independent steps, each correct with probability p.
  P(task correct) = pᵏ = p⁵

  p = 0.40 (small model):  P(correct) = 0.40⁵ = 0.010 ≈ 0%
  p = 0.60 (medium model): P(correct) = 0.60⁵ = 0.078 ≈ 8%
  p = 0.80 (large model):  P(correct) = 0.80⁵ = 0.328 ≈ 33%
  p = 0.95 (huge model):   P(correct) = 0.95⁵ = 0.774 ≈ 77%

  ON A GRAPH OF EXACT-MATCH ACCURACY vs MODEL SIZE:
  This looks like SUDDEN EMERGENCE (0% → 33% appears abrupt)!

  But on a graph of PER-STEP accuracy:
  0.40 → 0.60 → 0.80 → 0.95  ← smooth, predictable

  ┌─────────────────────────────────────────────────────┐
  │  Metric:           "smooth" vs "sharp"              │
  │  ─────────────────────────────────────────────────  │
  │  Bits per token:   always smooth (linear metric)    │
  │  Exact-match:      sharp transitions (non-linear)   │
  │  BLEU score:       somewhat smooth                  │
  │  Pass@1 (code):    sharp  (must be 100% correct)    │
  └─────────────────────────────────────────────────────┘
```

## 4. Mechanistic Emergence: Induction Heads

```
INDUCTION HEADS are attention heads that learn "if pattern [A][B...A] was seen,
predict [B] next". This is the foundation of in-context learning.

EMERGENCE OF INDUCTION HEADS:

Model size:        1-2 layers         3+ layers
                     ↓                    ↓
Attention head:   Random patterns     Induction pattern

  Small (2-layer):    Head 1: ████ (fuzzy, no clear pattern)
  
  Medium (4-layer):   Head 1: ██   (diagonal = prev token)
                      Head 2:     ██ (induction = prev A→B)

  ICL Performance jump:
    0.0 ─────────────────  ·  ·  ·  ·  ·  ·  ·   (in-context learning)
    0.5                                    (sudden jump when induction heads form)

TRANSITION is ABRUPT: the head suddenly "snaps" into the induction pattern
at a certain scale due to the loss landscape structure.
This is a true phase transition, not a metric artifact.
```

## 5. Practical Implications for LLM Development

```
CAPABILITIES ROADMAP (approximate thresholds):

  10M params:   │ Language patterns, basic grammar
  100M params:  │ Simple QA, basic translation
  1B params:    │ In-context learning (few-shot)
  7B params:    │ Instruction following (with SFT/RLHF)
                │ Basic code generation
                │ Simple reasoning
  30B params:   │ Consistent multi-step reasoning
  70B params:   │ Complex instruction following
                │ Long-form generation quality
  100B+ params: │ Chain-of-thought reasoning
                │ Complex math (with CoT)
  400B+ params: │ Advanced coding, complex reasoning
                │ Near-human performance on many tasks

  NOTE: Fine-tuning can shift these thresholds significantly.
  LLaMA-3 8B-Instruct achieves some capabilities expected of 70B base models.

LESSON FOR PRACTITIONERS:
  ┌──────────────────────────────────────────────────────┐
  │  When evaluating a base model at scale N:            │
  │  1. Use CONTINUOUS metrics (bits/token, not just %)  │
  │  2. Test capabilities AT the threshold (±1 order)    │
  │  3. Few-shot prompting can reveal latent capabilities │
  │  4. SFT/RLHF can unlock capabilities present but     │
  │     not accessible via zero-shot prompting           │
  └──────────────────────────────────────────────────────┘
```

---

## Exercises

1. Explain how a task requiring k=5 independent correct steps would appear to "emerge" even if per-step accuracy grows smoothly with scale.
2. The Chain-of-Thought (CoT) prompting technique appears to work only in models >100B. Propose a mechanistic explanation for why smaller models cannot benefit from CoT.
3. Design an experiment to determine whether a capability is "truly emergent" vs an artifact of a discrete metric. What would a continuous version of the metric look like?
