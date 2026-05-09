# 01 — Greedy Decoding and Beam Search

## 1. Greedy Decoding — Formal Definition

```
GREEDY STRATEGY:
  At each step t, select the single token with highest probability:

  wₜ* = argmax_{k ∈ V} P_θ(k | w₁, ..., wₜ₋₁)
       = argmax_{k ∈ V} zₖ          ← argmax of logits = argmax of softmax

  NO sampling, NO randomness. Fully deterministic.

SEQUENCE SCORE (log-probability):
  score(w₁...wₙ) = ∑_{t=1}^{n} log P_θ(wₜ | w<t)

Greedy maximises EACH TERM independently, not the sum.
This is the source of greedy's fundamental limitation.
```

## 2. Why Greedy Fails — Counterexample

```
SMALL VOCABULARY EXAMPLE:
  V = {A, B, C},  Greedy vs Optimal

  Step 1:
    P(A|BOS) = 0.55  ← greedy picks A
    P(B|BOS) = 0.30
    P(C|BOS) = 0.15

  Step 2 (given A was chosen):
    P(A|A) = 0.10
    P(B|A) = 0.20
    P(C|A) = 0.70  ← greedy picks C (but sequence is now AC)

  Step 2 (if B had been chosen):
    P(A|B) = 0.05
    P(B|B) = 0.10
    P(C|B) = 0.85  ← sequence BC has score log(0.30)+log(0.85)

COMPARISON of complete sequences:
  Greedy path: AC  → score = log(0.55) + log(0.70) = −0.597 + (−0.357) = −0.954
  Better path: BC  → score = log(0.30) + log(0.85) = −1.204 + (−0.163) = −1.367  (worse)
  Another:     AB  → score = log(0.30) + log(0.10) = −1.204 + (−2.303) = −3.507

  Greedy IS optimal here, but consider:
  
  Step 1:
    P(A|BOS) = 0.40  ← greedy picks A
    P(B|BOS) = 0.35
    P(C|BOS) = 0.25

  Step 2 given A:  P(C|A)=0.40  → greedy path: AC, score = log(0.40)+log(0.40)=−1.833
  Step 2 given B:  P(B|B)=0.95  → optimal path: BB, score = log(0.35)+log(0.95)=−1.100 ← BETTER!
  
  Greedy missed BB (score 0.65) because step 1 chose A (0.40 > 0.35).
```

## 3. Beam Search — Complete Algorithm

```
ALGORITHM (beam width B):

  INITIALISE:
    beams = [(score=0, sequence=[BOS])]   ← single initial beam

  FOR t = 1, 2, 3, ...:
    candidates = []
    
    FOR each (score_h, sequence_h) in beams:
      FOR each token k in V:
        new_score = score_h + log P_θ(k | sequence_h)
        new_seq   = sequence_h + [k]
        candidates.append((new_score, new_seq))
    
    candidates.sort(key=lambda x: x.score, DESCENDING)
    
    beams_new = []
    FOR each (score, seq) in candidates[:∞]:   ← consider all
      IF seq ends with EOS:
        completed.append((score, seq))         ← move to completed
      ELSE:
        beams_new.append((score, seq))
      IF len(beams_new) == B: BREAK             ← keep only top B
    
    beams = beams_new

  RETURN best sequence in completed (or if none, best in beams)

COMPLEXITY:
  Per step: B × |V| candidate scores computed
  Steps until EOS (average length L)
  Total: B × |V| × L   model forward passes... but wait!
  
  With KV cache: each beam is an independent sequence.
  KV cache is maintained separately for each beam.
  → B model forward passes per step (not B×|V|!)
  → |V| is handled by the logits vector, not separate forward passes
```

## 4. Length Penalty — Why and How

```
PROBLEM: Beam search score = ∑ log P(wₜ) is a SUM.
  Each term is negative: log P(wₜ) ≤ 0.
  Longer sequences accumulate more negative terms → lower score!
  
  ┌──────────────────────────────────────────────────────────┐
  │  "yes" (1 token):     score = log(0.3) = −1.2            │
  │  "yes it is" (3 tok): score = log(0.3)+log(0.8)+log(0.9) │
  │                             = −1.2 − 0.22 − 0.10 = −1.52 │
  │                                                           │
  │  Beam search prefers "yes" over "yes it is"               │
  │  even though the full answer is more informative!         │
  └──────────────────────────────────────────────────────────┘

LENGTH NORMALISATION:
  score_normalised(w₁...wₙ) = (1/n^α) × ∑_t log P(wₜ | w<t)
  
  α = 0: no normalisation (raw log-prob, biased toward short)
  α = 1: divide by length (penalise short AND long equally)
  α ∈ (0,1): partial normalisation (typical: α=0.6 in Google NMT)

CHOOSING α:
  Consider two sequences of length 3 and 5:
    Short: sum log P = −3.0,  normalised = −3.0/3^{0.6} = −3.0/1.93 = −1.55
    Long:  sum log P = −4.5,  normalised = −4.5/5^{0.6} = −4.5/2.63 = −1.71

  At α=0.6:  short wins   (−1.55 > −1.71)
  If long had sum=−4.0:   −4.0/2.63 = −1.52  long wins  ← depends on content quality

  EMPIRICALLY: α=0.6–0.8 works well for translation/summarisation.
```

## 5. When to Use Each Decoding Strategy

```
TASK COMPARISON:

Task                    │ Recommended │ Why
────────────────────────┼─────────────┼──────────────────────────────────────────
Machine translation     │ Beam B=4    │ Single correct answer, quality matters
Code generation         │ Beam B=4    │ Syntax must be correct, deterministic preferred
Mathematical reasoning  │ Greedy      │ Temperature=0 or beam B=1 (single correct answer)
Story writing           │ Sampling    │ Diversity needed, multiple good answers exist
Chatbot response        │ T+top-p     │ Natural, varied, contextually appropriate
Summarisation           │ Beam B=4    │ Content coverage matters, some creativity OK
Scientific QA           │ Greedy/T=0  │ Factual accuracy, reproducible

MATHEMATICAL COMPARISON:

  GREEDY = Beam B=1:
    Maximize per-step log prob.
    Optimal at each step but NOT globally optimal.
    O(1) overhead per token beyond base inference.

  BEAM B:
    Maintains B partial sequences simultaneously.
    Provably better global solution than greedy (within B candidates).
    Memory: B × (KV cache per sequence)
    
    For LLaMA-3 8B, B=4, L=500 tokens:
    Extra memory vs greedy: 3 × (500 × 128 KB) ≈ 192 MB  (manageable)

  SAMPLING (next section):
    Does NOT maximise probability.
    Samples from P_θ(·|context) with modifications.
    Non-deterministic: different output each run.
```

---

## Exercises

1. Show that beam search with B=1 is identical to greedy decoding (prove they produce the same output).
2. For a vocabulary of |V|=50K and sequence length L=100, how many total sequences does beam search (B=5) maintain at each step? Why is this tractable compared to full search?
3. Prove that the length-normalised score (1/n^α) × ∑ log P is equivalent to geometric mean of per-step probabilities raised to the power 1/α. What does this mean intuitively?
