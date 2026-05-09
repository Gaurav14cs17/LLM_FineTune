# Understanding Byte-Pair Encoding (BPE)

*A comprehensive guide to subword tokenization — from raw text to vocabulary construction*

---

**Byte-Pair Encoding (BPE)** is the tokenization algorithm used in GPT-2, GPT-3, GPT-4, LLaMA, and Mistral. It learns a vocabulary of subword units by iteratively merging the most frequent character pairs. This allows the model to handle both common words (as single tokens) and rare words (split into known subwords).

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Subword Tokenization?](#11-why-subword-tokenization)
   - [1.2 BPE vs Word vs Character](#12-bpe-vs-word-vs-character)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The BPE Algorithm](#2-the-bpe-algorithm)
   - [2.1 Initialisation](#21-initialisation)
   - [2.2 Merge Operations](#22-merge-operations)
   - [2.3 Full Example](#23-full-example)
3. [Encoding New Text](#3-encoding-new-text)
   - [3.1 Tokenization at Inference](#31-tokenization-at-inference)
4. [Byte-Level BPE](#4-byte-level-bpe)
   - [4.1 Why Bytes?](#41-why-bytes)
5. [Summary](#5-summary)
   - [5.1 Formulas Quick Reference](#51-formulas-quick-reference)
   - [5.2 Common Mistakes](#52-common-mistakes)
6. [Exercises](#6-exercises)

---

## 1. Overview

### 1.1 Why Subword Tokenization?

```
┌─────────────────────────────────────────────────────────────┐
│  VOCABULARY TRADE-OFF                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WORD-LEVEL (e.g., "play", "plays", "playing"):             │
│  + Short sequences (1 token per word)                       │
│  - Large vocabulary → huge embedding table                  │
│  - OOV: "ChatGPT" not in 1950s vocabulary → [UNK]          │
│                                                             │
│  CHAR-LEVEL (e.g., "p", "l", "a", "y"):                     │
│  + No OOV — can represent ANY text                          │
│  - Very long sequences — poor context efficiency            │
│  - 4096 context window = only ~820 words!                   │
│                                                             │
│  SUBWORD BPE (e.g., "play", "##ing", "##s"):                │
│  + Short sequences for common words                         │
│  + No OOV — rare words split into known subwords            │
│  + Vocabulary size manageable (~30K-128K)                   │
│  ← BEST TRADE-OFF                                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 BPE vs Word vs Character

| Method | Vocab size | Avg tokens per word | OOV handling |
|--------|-----------|---------------------|-------------|
| Word-level | 50K–300K | 1.0 | ❌ fails on new words |
| Character | 100–500 | 4–7 | ✅ perfect | 
| BPE (GPT-4) | 100,257 | 1.2–1.5 | ✅ always decodable |

### 1.3 Pipeline Summary

```
TRAINING (learn vocabulary from corpus):
  Text corpus
      ↓
  Initialise: characters as base vocabulary
      ↓
  Count: frequency of all adjacent symbol pairs
      ↓
  Merge: most frequent pair into a new symbol
      ↓
  Repeat until vocabulary size = V_target
      ↓
  Output: list of V_target - V_base merge rules

ENCODING (tokenize new text):
  Input text → apply pre-tokenization (split on spaces/punctuation)
      ↓
  Split each word into characters
      ↓
  Apply ALL merge rules in order of when they were learned
      ↓
  Output: sequence of token IDs
```

---

## 2. The BPE Algorithm

### 2.1 Initialisation

```
CORPUS (toy example):
  "low low low low low lower lower newest newest widest"

PRE-TOKENIZATION (split on whitespace + add end marker):
  "low" → l o w </w>   (appears 5 times)
  "lower" → l o w e r </w>   (appears 2 times)
  "newest" → n e w e s t </w>   (appears 6 times)
  "widest" → w i d e s t </w>   (appears 3 times)

INITIAL VOCABULARY (all unique characters):
  {l, o, w, e, r, n, s, t, i, d, </w>}   ← 11 symbols

INITIAL SYMBOL FREQUENCIES:
  l o w </w>   × 5      l o w e r </w>   × 2
  n e w e s t </w>   × 6      w i d e s t </w>   × 3
```

### 2.2 Merge Operations

```
COUNT ALL ADJACENT PAIRS:
  From "l o w </w>" × 5:    (l,o)=5, (o,w)=5, (w,</w>)=5
  From "l o w e r </w>" × 2: (l,o)+=2, (o,w)+=2, (w,e)=2, (e,r)=2, (r,</w>)=2
  From "n e w e s t </w>" × 6: (n,e)=6, (e,w)=6, (w,e)+=6, (e,s)=6, (s,t)=6+3=9...
  ...

FULL PAIR COUNTS:
  (e, s): 6+3 = 9
  (s, t): 6+3 = 9
  (e, w): 6    (l, o): 7    (o, w): 7    (w,</w>): 5

MERGE STEP 1: pick (e, s) or (s, t) — tie, pick first alphabetically:
  Merge (e, s) → "es":
  "n e w es t </w>" × 6    "w i d es t </w>" × 3
  New symbol: "es"

MERGE STEP 2: count again. Now "est" = (es, t) appears 6+3=9 times.
  Merge (es, t) → "est":
  "n e w est </w>" × 6    "w i d est </w>" × 3

MERGE STEP 3: (l, o) = 7.
  Merge (l, o) → "lo":
  "lo w </w>" × 5    "lo w e r </w>" × 2

MERGE STEP 4: (lo, w) = 7.
  Merge (lo, w) → "low":
  "low </w>" × 5    "low e r </w>" × 2

MERGE STEP 5: (e, w) = 6.
  Merge (e, w) → "ew":
  "n ew est </w>" × 6

CONTINUE until desired vocabulary size V is reached.
```

### 2.3 Full Example

```
After 5 merge steps:
  VOCABULARY: {l,o,w,e,r,n,s,t,i,d,</w>, es,est,lo,low,ew}

TOKENIZATION RESULT:
  "low"    → ["low", "</w>"]     → token IDs [14, 10]
  "lower"  → ["low", "e", "r", "</w>"]   → [14, 4, 5, 10]
  "newest" → ["n", "ew", "est", "</w>"]  → [6, 15, 12, 10]
  "widest" → ["w", "i", "d", "est", "</w>"] → [3, 9, 8, 12, 10]
  "newest" uses 4 tokens vs 7 characters — compression!

MERGE HISTORY (the vocabulary file):
  MERGE 1: e s → es
  MERGE 2: es t → est
  MERGE 3: l o → lo
  MERGE 4: lo w → low
  MERGE 5: e w → ew
  ...
```

---

## 3. Encoding New Text

### 3.1 Tokenization at Inference

```
NEW WORD: "lowest" (not in training data)

STEP 1: Start from characters:
  l o w e s t </w>

STEP 2: Apply merge rules IN ORDER:

  Rule 1 (e s → es): l o w es t </w>
  Rule 2 (es t → est): l o w est </w>
  Rule 3 (l o → lo): lo w est </w>
  Rule 4 (lo w → low): low est </w>
  [no more applicable rules]

RESULT: ["low", "est", "</w>"]  → 3 tokens

Vocabulary handles "lowest" perfectly even though it was never in training!
It's split into meaningful subwords "low" + "est" (superlative suffix).

UNKNOWN CHARACTERS:
  If a character is not in the initial byte vocabulary:
    → GPT-2 (byte-level BPE): IMPOSSIBLE — all 256 bytes are in vocab!
    → Char-level BPE: falls back to [UNK]
```

---

## 4. Byte-Level BPE

### 4.1 Why Bytes?

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: CHAR-LEVEL BPE FAILS ON UNICODE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Unicode has 140,000+ characters.                           │
│  Initial char vocabulary would be 140K+ tokens.             │
│  Many rare characters appear too infrequently to merge.     │
│  → "佐藤" (Japanese) might never be merged → too long        │
│                                                             │
│  BYTE-LEVEL BPE (GPT-2, RoBERTa, LLaMA):                  │
│  Initial vocabulary = 256 bytes (ALL possible byte values)  │
│  EVERY unicode character = 1-4 UTF-8 bytes = ALWAYS in vocab│
│                                                             │
│  "佐" → UTF-8 bytes: 0xE4 0xBD 0x90                         │
│       → Byte tokens: [228, 189, 144] (decimal)              │
│       → May merge to a single "佐" token if frequent enough │
│                                                             │
│  GUARANTEED: Any text can always be tokenized, no [UNK]!   │
└─────────────────────────────────────────────────────────────┘

BYTE-LEVEL BPE VOCABULARY (GPT-2/GPT-4):
  - 256 base byte tokens (0–255)
  - + merges learned until |V| = 50,257 (GPT-2) or 100,257 (cl100k)
  - = complete lossless tokenizer for ANY text in ANY language
```

---

## 5. Summary

### 5.1 Formulas Quick Reference

**Pair frequency:**

```
f(a, b) = number of times symbol a is immediately followed by b
          summed over all words, weighted by word frequency
```

**Greedy merge:**

```
At each step: merge (a*, b*) = argmax_{(a,b)} f(a, b)
```

**BPE vocabulary size:**

```
|V_final| = |V_init| + K_merges
  V_init: base character/byte vocabulary
  K_merges: number of merge operations performed
```

**Sequence length after tokenization:**

```
Expected tokens per word ≈ 1 + (original chars - 1) × (1 - compression_rate)
For English, GPT-4 BPE: ~1.3 tokens/word average
For code: ~3–4 tokens/word (more symbols, fewer merges)
For CJK: ~1.5–2 tokens/character
```

| Algorithm | Base vocab | Merge | Languages |
|-----------|-----------|-------|-----------|
| Character BPE | Unique chars | Pairs of chars | Poor on unseen chars |
| Byte BPE | 256 bytes | Pairs of bytes | Perfect for all Unicode |
| WordPiece (BERT) | Chars | Maximise likelihood | Word-internal only |
| Unigram LM | Large vocab | Prune by likelihood | Principled, flexible |

### 5.2 Common Mistakes

```
❌ WRONG: BPE tokenisation is deterministic but slow — must recompute each time
✓ RIGHT:  The tokenizer is precomputed and stored as a set of merge rules.
          Encoding new text is O(n × log n) where n is text length.

❌ WRONG: BPE creates the same tokenization regardless of context
✓ RIGHT:  BPE is indeed deterministic (given a fixed merge table).
          Unlike Unigram LM, BPE does NOT do stochastic tokenization.
          SentencePiece supports stochastic BPE for data augmentation.

❌ WRONG: A larger vocabulary is always better
✓ RIGHT:  Larger vocabulary → shorter sequences → more context, but also
          larger embedding table (memory), sparser gradients per token,
          and more model parameters (O(|V| × d)).
          There's an optimal vocabulary size for each model size and language.

❌ WRONG: BPE can handle all languages equally well with the same vocab
✓ RIGHT:  BPE trained on English-dominant corpora tokenizes English efficiently
          (~1.3 tokens/word) but non-English text much less efficiently
          (~4-8 tokens/word for low-resource languages).
          Multilingual models need balanced corpora for training the tokenizer.
```

---

## 6. Exercises

1. **Merge Simulation**: Given the vocabulary `{a, b, c, ab}` and corpus "ab ab ab a b a b c c", run 2 BPE merge steps. Show the pair frequencies and which merges are chosen.

2. **Tokenization**: With merge rules: (e,s)→es, (es,t)→est, (l,o)→lo, (lo,w)→low, tokenize the word "lowest". Show each rule application step.

3. **Vocabulary Size Trade-off**: A model has d=1024, vocabulary options: |V|=10K, 50K, 100K. Compute the size of the embedding matrix (in bytes, BF16) for each. What's the overhead of going from 50K to 100K?

4. **Byte-Level Encoding**: The Japanese character "猫" (cat) is encoded in UTF-8 as bytes 0xE7 0x8C 0xAB (decimal: 231, 140, 171). In byte-level BPE with no merges for this sequence, how many tokens does "猫" become? What if it appears frequently enough to be merged into one token?
