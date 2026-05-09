# 03 — SentencePiece & Unigram Language Model Tokenizer

## 1. SentencePiece: Language-Agnostic Tokenization

```
TRADITIONAL pipeline (breaks for many languages):
  Raw text
     │
     ▼
  Pre-tokenize (split on spaces, punctuation)   ← FAILS for Chinese, Japanese, Arabic
     │
     ▼
  Apply BPE/WordPiece
     │
     ▼
  Tokens

SENTENCEPIECE pipeline:
  Raw text  ──────────────────────────►  Tokens
  (treats entire byte stream as input, no language-specific rules)

WORD BOUNDARY ENCODING with ▁ (U+2581):
  Input:  "Hello world"
  Output: ["▁Hello", "▁world"]     ← ▁ marks beginning of a word/space

  Input:  "Hello" (in middle of sentence, after space was removed)
  Output: ["▁Hello"]               ← same token regardless of surrounding context

REVERSIBILITY TEST:
  ["▁Hello", "▁world"] → join → "▁Hello▁world" → replace ▁ with space → "Hello world" ✓
  ["▁Hello", "world"]  → join → "▁Helloworld"  → replace ▁ with space → "Helloworld"  (different!)
  → The ▁ prefix is ESSENTIAL for lossless reconstruction
```

## 2. Unigram LM: Start Big, Prune Down

```
BPE (bottom-up):                  Unigram LM (top-down):
  Start with characters             Start with ALL substrings (large vocab)
  Iteratively ADD merges            Iteratively REMOVE tokens
  ┌──────────┐                      ┌──────────────────────────────────┐
  │ a b c d  │ → add "ab"           │ a,b,c,d,ab,bc,abc,bcd,abcd,...  │ → remove least useful
  └──────────┘                      └──────────────────────────────────┘

Unigram guarantees: maximises likelihood of the corpus under the current vocabulary.
```

## 3. Segmentation as a Probabilistic Model

```
WORD: "newest"
POSSIBLE SEGMENTATIONS (all substrings in vocabulary):
  [1] ["newest"]          → log P = log P("newest")
  [2] ["new", "est"]      → log P = log P("new") + log P("est")
  [3] ["n", "ew", "est"]  → log P = log P("n") + log P("ew") + log P("est")
  [4] ["ne", "west"]      → log P = log P("ne") + log P("west")
  ...many more...

SEGMENTATION LATTICE for "newest":
  n ─► ne ─► new ─► newe ─► newes ─► newest
  │    │      │       │        │
  │    └──►  ew ─►  ews ─►  ewes ─►  ewest
  │           │       │
  │           └──►   est ─► ...
  │                   │
  └──────────────────►│
                    (every path = one segmentation)

BEST SEGMENTATION: Viterbi algorithm finds path with maximum ∑ log P(piece)

  best[0] = 0
  For j = 1 to len("newest"):
    For each subword x[i:j] ∈ vocab:
      best[j] = max(best[j], best[i] + log P(x[i:j]))
  
  Backtrack from best[6] to recover the segmentation.
```

## 4. Unigram Training: EM + Pruning

```
ITERATION:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  Large vocab V  ──► E-step: for each word w,        │
  │                            compute expected counts  │
  │                            of each piece using      │
  │                            forward-backward alg.    │
  │                     ↓                               │
  │                     M-step: re-estimate P(piece)    │
  │                            = expected_count(piece)  │
  │                              / total_expected_count │
  │                     ↓                               │
  │                     PRUNE: remove bottom η%         │
  │                            of pieces by |Δloss|     │
  │                     ↓                               │
  │                  Repeat until |V| = target          │
  └─────────────────────────────────────────────────────┘

|Δloss(t)| = how much the corpus log-likelihood drops if token t is removed.
Keep tokens with LARGE |Δloss| (removing them would hurt a lot).
Remove tokens with SMALL |Δloss| (they are redundant with other tokens).
```

## 5. Stochastic Tokenization (Training-Time Regularisation)

```
DETERMINISTIC (inference):       STOCHASTIC (training):
  best path through lattice        sample a path ∝ probability

  "newest"                         Iteration 1: ["new", "est"]
     │                             Iteration 2: ["newest"]        ← same word!
  ["new", "est"]                   Iteration 3: ["n", "ew", "est"]

This acts as DATA AUGMENTATION:
  The model sees the same word with different tokenizations
  → learns more robust representations
  → less sensitive to exact tokenization boundaries
```

---

## Exercises

1. Why does SentencePiece's ▁ prefix make tokenization fully reversible?
2. In the Unigram pruning step, why keep tokens with **large** |Δloss| rather than small?
3. Stochastic tokenization samples from P(s|x). What regularisation effect does this have?
   Compare to subword regularization in BPE (BPE-dropout).
