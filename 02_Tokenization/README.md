# Chapter 02 — Tokenization

Tokenization converts raw Unicode text into integer IDs that the model processes.
The choice of tokenizer affects vocabulary coverage, sequence length, and cross-lingual transfer.

---

## Sub-topics

| # | Folder | Algorithm | Used by |
|---|--------|-----------|---------|
| 1 | `01_BPE/` | Byte-Pair Encoding | GPT-2, GPT-4, LLaMA |
| 2 | `02_WordPiece/` | WordPiece | BERT, mBERT |
| 3 | `03_SentencePiece_Unigram/` | Unigram Language Model | LLaMA-3, mT5 |

---

## Key Concepts

**Vocabulary size trade-off:**
```
Small |V| → long sequences, more context needed, cheaper embedding table
Large |V| → short sequences, better coverage, huge embedding table
```

**Fertility** (tokens per word) measures how efficiently a tokenizer encodes a language.

**BPE merge score:**
```
score(A, B) = freq(AB) / (freq(A) · freq(B))   [in WordPiece]
score(A, B) = freq(AB)                          [in BPE]
```

**Unigram LM log-likelihood for segmentation:**
```
log P(x) = ∑_{i} log P(xᵢ)   over the best segmentation
```

**SentencePiece** treats the input as a raw byte stream, enabling language-agnostic tokenization
with a configurable vocabulary size and a special ▁ (U+2581) character for word boundaries.
