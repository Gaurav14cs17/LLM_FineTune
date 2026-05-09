# 02 — WordPiece Tokenisation

## 1. BPE vs WordPiece — Core Difference

```
BPE OBJECTIVE:
  Greedily merge the MOST FREQUENT pair at each step.
  Optimises: raw bigram count  (not a probabilistic objective)

WORDPIECE OBJECTIVE:
  Merge the pair that MAXIMISES the language model likelihood.
  Objective: max ∑_{w ∈ corpus} log P(w | vocabulary V)
  
  This is a principled probabilistic criterion:
    Choose the merge that most increases P(corpus).
    Equivalent to: choose the merge that gives the largest information gain.
```

## 2. The WordPiece Scoring Function — Derivation

```
CURRENT STATE: vocabulary V with a unigram language model.
  P(t) = count(t) / total_tokens   for each token t ∈ V

CANDIDATE MERGE: (x, y) → xy

EFFECT on corpus log-likelihood:
  Before merge: each occurrence of [x, y] contributes log P(x) + log P(y)
  After merge:  each occurrence of [xy]  contributes log P(xy)
  
  CHANGE in log-likelihood per occurrence of (x,y):
    ΔLL = log P(xy) − log P(x) − log P(y)
        = log [P(xy) / (P(x) × P(y))]
        = log [PMI(x, y)]

  where PMI = Pointwise Mutual Information.

WORDPIECE SCORE for pair (x, y):
  ┌───────────────────────────────────────────────────────────────┐
  │                   count(xy)                                   │
  │  score(x, y) = ─────────────────────────                     │
  │                   count(x) × count(y)                        │
  └───────────────────────────────────────────────────────────────┘

  This is EXACTLY exp(PMI) (pointwise mutual information exponentiated):
  
  PMI(x,y) = log P(x,y) − log P(x) − log P(y)
           = log [count(xy)/N] − log [count(x)/N] − log [count(y)/N]
           = log [count(xy) / (count(x) × count(y) / N)]
  
  score(x,y) = count(xy) / (count(x) × count(y)) = N × exp(PMI)

INFORMATION THEORY INTERPRETATION:
  PMI(x,y) > 0: x and y co-occur MORE than expected by chance → good merge candidate
  PMI(x,y) = 0: x and y co-occur exactly as expected → neutral
  PMI(x,y) < 0: x and y co-occur LESS than expected → bad merge candidate
  
  WordPiece selects merges with the highest PMI → maximises information sharing.
```

## 3. WordPiece vs BPE — Concrete Example

```
CORPUS: "aaab" × 5, "aaac" × 2
Initial vocab: {a, b, c}

BPE:
  Pair counts: (a,a)=21, (a,b)=5, (a,c)=2
  Best pair: (a,a) [count=21] → merge → "aa"
  New corpus: "aab" ×5, "aac" ×2
  
  Pair counts: (aa,b)=5, (aa,c)=2, (a,a)=0, (a,b)=0
  Best pair: (aa,b) → merge → "aab"
  ...

WORDPIECE:
  Scores = count(pair) / (count(a) × count(a or b or c)):
    score(a,a) = 21 / (count(a)×count(a))
               count(a) as FIRST element = 21 occurrences
               count(a) as SECOND element = 21 occurrences
               score(a,a) = 21/(21×21) = 0.048

    score(a,b) = 5 / (5×5) = 0.20   ← higher PMI!
    score(a,c) = 2 / (5×2) = 0.20   ← same PMI

  WordPiece picks (a,b) or (a,c) first!
  → Merges the RARER but more SPECIFIC pairs.
  → BPE merges the most common pair (a,a) first.

CONSEQUENCE:
  WordPiece vocabulary captures "ab" and "ac" as units (meaningful subwords).
  BPE vocabulary captures "aa" (which just means "two a's in a row", less meaningful).
  
  WordPiece tends to produce more SEMANTICALLY meaningful subwords.
  BPE tends to produce more FREQUENT subwords.
```

## 4. BERT's WordPiece Implementation Details

```
BERT vocabulary construction:
  1. Start with characters + special tokens ([UNK],[CLS],[SEP],[MASK],[PAD])
  2. Apply WordPiece training on English Wikipedia + BookCorpus
  3. Target |V| = 30,522 (BERT-base) or 32,000 (various models)

BERT-SPECIFIC CONVENTIONS:
  Continuation subwords marked with "##":
    "unhappiness" → ["un", "##happi", "##ness"]
    "##happi" signals "this subword is a continuation of the previous"
    
  LLaMA/GPT convention: leading space on first subword:
    "unhappiness" → ["▁un", "happin", "ess"]  (▁ marks word-start)

COMPARISON:
  Method      │ Marker      │ "running"           │ "unhappy"
  ────────────┼─────────────┼─────────────────────┼─────────────────
  BPE (GPT)   │ ▁ (leading) │ ["▁run","ning"]     │ ["▁un","happy"]
  WordPiece   │ ## (suffix) │ ["run","##ning"]    │ ["un","##happy"]
  
  Semantically equivalent, just different notation.
```

## 5. Unigram Language Model — The Principled Alternative

```
WORDPIECE IS STILL GREEDY (bottom-up).
UNIGRAM LM (Kudo 2018) is the fully principled version:

  Instead of greedily building up from characters,
  start with a LARGE vocabulary and PRUNE it.

ALGORITHM:
  INITIALISE: V = top 10|V| most frequent substrings
  REPEAT until |V| = target:
    1. E-step: for each word w, compute P(w | V) using Viterbi:
               P(w) = max_{segmentation} ∏_{t in seg} P(t)
    2. M-step: recompute P(t) for each token t:
               P(t) = ∑_w count(w) × (P(t ∈ optimal_seg(w) | w))
               (expected frequency under current segmentation)
    3. Compute marginal log-likelihood: L(V) = ∑_w count(w) × log P(w | V)
    4. For each token t, compute L(V \ {t}) − L(V)  (loss from removing t)
    5. Remove the 10-20% of tokens with smallest loss (most redundant)
  
  This is the EM ALGORITHM applied to tokenisation!

MATHEMATICAL PROPERTY:
  Unigram LM maximises:
    L(V) = ∑_w count(w) × log P(w | V)
  
  This is exactly the corpus log-likelihood under the unigram model.
  
  Compare to BPE: maximises compression (greedy byte-pair merger)
  Compare to WordPiece: maximises pairwise PMI (greedy)
  Unigram LM: maximises FULL corpus likelihood (principled, global)
  
  → Unigram LM is theoretically superior (minimises redundancy globally)
  → BPE/WordPiece are faster but use heuristic greedy criteria
  
  SentencePiece (used by LLaMA) implements both BPE and Unigram LM.
```

---

## Exercises

1. Compute the WordPiece score for pairs (a,b) and (a,c) in the corpus "aa b" ×10, "a b" ×5, "a c" ×3. Show that WordPiece merges (a,b) first (highest score).
2. Prove algebraically that WordPiece score = count(xy) / (count(x) × count(y)) is proportional to exp(PMI(x,y)) up to a constant factor.
3. Explain why Unigram LM is "globally optimal" while BPE and WordPiece are "locally greedy". What computational complexity makes global optimisation harder for tokenisation?
