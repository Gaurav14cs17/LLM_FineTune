# 02 — Sampling Strategies

## 1. The Problem with Pure Sampling

```
MODEL OUTPUT logits for "The cat sat on the ___":
  "mat" :  z = 4.2  →  P = 0.42
  "floor": z = 3.1  →  P = 0.18
  "roof" : z = 2.4  →  P = 0.09
  ...
  "moon" : z = -2.1 →  P = 0.0008
  ...50K tokens, long tail of very unlikely choices...

PURE SAMPLING: sample from the full distribution
  99% of the time: sensible output ("mat", "floor", "chair", ...)
  1% of the time:  random garbage ("moon", "desk", "airplane", ...)

After 100 tokens of generation:
  Prob(at least one bad token) = 1 − 0.99^100 ≈ 63%  ← very likely to go off-rails

SOLUTION: Truncate the tail before sampling.
```

## 2. Temperature Scaling

```
EFFECT of temperature T on the distribution:

logits z = [4.2, 3.1, 2.4, 1.0, -2.1]

T = 0.5 (sharper):   z/T = [8.4, 6.2, 4.8, 2.0, -4.2]
  softmax →  P = [0.78, 0.15, 0.05, 0.01, ≈0]  ← very concentrated

T = 1.0 (original):
  softmax →  P = [0.42, 0.18, 0.09, 0.02, 0.0008]  ← original

T = 2.0 (flatter):   z/T = [2.1, 1.55, 1.2, 0.5, -1.05]
  softmax →  P = [0.25, 0.17, 0.12, 0.06, 0.01]  ← more uniform

VISUALISED:

T=0.5 │ ████████████████████ (first token dominates)
      │ ████
      │ █
      │ 

T=1.0 │ ██████████ (balanced)
      │ █████
      │ ███
      │ ██

T=2.0 │ ██████ (flatter)
      │ █████
      │ ████
      │ ███

T→0   = greedy (deterministic argmax)
T=1   = original model distribution
T→∞   = uniform random (all tokens equally likely)
```

## 3. Top-k Sampling

```
IDEA: Only consider the top k most probable tokens.

k=5 example (continuing from "The cat sat on the"):
  Vocabulary: 50,000 tokens
  
  STEP 1: Sort by probability (descending)
    "mat" :  P = 0.42   ← keep
    "floor": P = 0.18   ← keep
    "roof" : P = 0.09   ← keep
    "ground":P = 0.07   ← keep
    "table": P = 0.05   ← keep
    "wall" : P = 0.03   CUT HERE (k=5)
    "moon" : P = 0.0008 ← CUT
    ...49,994 more tokens: CUT
  
  STEP 2: Renormalise over top k
    Sum top-5: 0.42+0.18+0.09+0.07+0.05 = 0.81
    New probs: [0.52, 0.22, 0.11, 0.09, 0.06]

  STEP 3: Sample from renormalised distribution

PROBLEM with fixed k:
  "The answer is ___"          "Tell me a story about ___"
  P("42")   = 0.80             P("a")     = 0.05   (many valid continuations)
  P("yes")  = 0.07             P("the")   = 0.04
  P("no")   = 0.05             P("an")    = 0.04
  ...                          ...

  k=50 is too many for first case   k=50 is too few for second case
  → k should be ADAPTIVE!
```

## 4. Top-p (Nucleus) Sampling

```
IDEA: Include the smallest set of tokens covering p of the probability mass.

p = 0.9 example:

CASE 1: "The answer is ___" (confident context)
  "42" :   P = 0.80  → cumsum = 0.80  > 0.9? No, but 0.80 alone?  keep
  "yes":   P = 0.07  → cumsum = 0.87  keep
  "no" :   P = 0.05  → cumsum = 0.92  ≥ 0.9? YES → STOP
  V_p = {"42", "yes", "no"}  (only 3 tokens!)

CASE 2: "Tell me a story about ___" (uncertain context)
  "a"  :   P = 0.05  → cumsum = 0.05  keep
  "the":   P = 0.04  → cumsum = 0.09  keep
  "an" :   P = 0.04  → cumsum = 0.13  keep
  ...keep adding...
  "my" :   P = 0.01  → cumsum = 0.85  keep
  "his":   P = 0.01  → cumsum = 0.90  ≥ 0.9? YES → STOP
  V_p = {50+ tokens}  ← adapts to uncertainty!

NUCLEUS SIZE adapts automatically:
  Confident context → small nucleus → less random
  Uncertain context → large nucleus → more diverse

Algorithm:
  1. Sort tokens by P descending: w₍₁₎, w₍₂₎, ...
  2. Find min V_p: {w₍₁₎,...,w₍ₖ₎} where ∑ᵢ P(w₍ᵢ₎) ≥ p
  3. Renormalise over V_p
  4. Sample
```

## 5. Min-p Sampling (2023)

```
IDEA: Exclude tokens with probability < min_p × P_max

min_p = 0.05 example:
  P_max = 0.42 ("mat")
  threshold = 0.05 × 0.42 = 0.021

  "mat" :  P = 0.42  ≥ 0.021 → keep
  "floor": P = 0.18  ≥ 0.021 → keep
  "roof" : P = 0.09  ≥ 0.021 → keep
  "ground":P = 0.07  ≥ 0.021 → keep
  "table": P = 0.05  ≥ 0.021 → keep
  "wall" : P = 0.03  ≥ 0.021 → keep
  "chair": P = 0.01  < 0.021 → CUT
  "moon" : P = 0.0008< 0.021 → CUT

ADVANTAGE over top-p: the threshold automatically scales with model confidence.
If P_max drops (uncertain context), threshold drops too → larger nucleus.
If P_max rises (confident context), threshold rises → tighter focus.
```

## 6. Comparing Strategies Visually

```
Distribution: P = [0.42, 0.18, 0.09, 0.07, 0.05, 0.03, 0.02, 0.01, ...]
                   mat   floor  roof  ground table wall chair  moon ...

Strategy:    Tokens kept (highlighted ★):
Full dist:   ★  ★  ★  ★  ★  ★  ★  ★  ... (all 50K)
Top-k=5:     ★  ★  ★  ★  ★  ─  ─  ─  ─  (fixed)
Top-p=0.9:   ★  ★  ★  ★  ★  ★  ─  ─  ─  (cumsum stops at ~0.9)
Min-p=0.05:  ★  ★  ★  ★  ★  ★  ─  ─  ─  (threshold = 0.021)
Greedy:      ★  ─  ─  ─  ─  ─  ─  ─  ─  (only argmax)
```

---

## Exercises

1. For logits z = [2, 1, 0, -1] and T=0.5, compute the temperature-scaled probabilities. Compare to T=1 and T=2.
2. For a distribution where 5 tokens cover 90% of the mass, how does top-p (p=0.9) compare to top-k (k=5)? When do they differ?
3. Show that min-p sampling with min_p=0 is equivalent to pure sampling.
