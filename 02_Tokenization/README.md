# Chapter 02 — Tokenization

Raw text must become a **sequence of discrete token IDs** before embeddings or attention apply. Tokenizers define the **alphabet** of the LM: vocabulary design trades off sequence length, rare-word coverage, multilingual behaviour, and table size for embeddings. This chapter compares **BPE**, **WordPiece**, and **Unigram/SentencePiece**—the three families behind most open- and closed-weight models since GPT-2 and BERT.

```
  Unicode string  ──►  NORMALISE (NFKC, lower, strip…)   [model-dependent]
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SUBWORD SEGMENTATION: merge / score / sample segments               │
  │     "unaffordability" ─► ["un","afford","ability"]   (illustrative)  │
  └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
   integer IDs  t_1,…,t_T     (often + specials: BOS, EOS, PAD, UNK)
        │
        └── vocab tables live in embedding layers (Chapter 03)
```

## Sub-chapters

**`01_BPE/` (Byte-Pair Encoding)**  
Starts from symbols (bytes or characters) and iteratively **merges** the most frequent pairs, growing a vocabulary file of merge rules. Used in GPT-2/3/4-style stacks; deterministic given a corpus and a target \(|\mathcal{V}|\). Explains **pieces** vs **words** and why byte-level BPE avoids UNK for arbitrary Unicode at the cost of longer sequences for non-Latin scripts.

**`02_WordPiece/`**  
BERT-family tokenizers score candidate merges using **normalized frequencies** (score \(\propto \frac{f(AB)}{f(A)f(B)}\)) and aim to maximise language-model likelihood of segmented text under a bigram assumption. Emphasises compatibility with **masked** objectives (whole-word masking tricks often depend on WordPiece boundaries).

**`03_SentencePiece_Unigram/`**  
The Unigram LM tokenizer treats segmentation as latent: it learns token **unigram** probabilities and finds a high-probability split via **Viterbi**-style dynamic programming. SentencePiece packages **pre-tokenisation** and vocabulary learning in one tool; the **SPM** model file is portable across languages when trained on raw sentences **without** space-presplitting (Unicode is the stream).

## Key formulas and concepts

**Fertility**

\[
\text{fertility} = \frac{\text{\# tokens}}{\text{\# words (reference)}}\,;
\]

useful for cross-tokenizer sequence-length planning.

**BPE merge objective (sketch)**  
Iteratively pick merge \((a,b)\) **maximising** co-occurrence frequency of the **new** symbol \(ab\) in the corpus, subject to vocabulary cap.

**WordPiece score (conceptual)**

\[
\mathrm{score}(a,b) = \frac{f(ab)}{f(a)\,f(b)}.
\]

**Unigram target** (segment \(x\) into \(\{x_i\}\))

\[
\max_{\mathrm{segmentation}} \sum_i \log p(x_i), \quad \sum_{w\in\mathcal{V}} p(w)=1.
\]

**Practical trade-off**

\[
|\mathcal{V}| \uparrow \Rightarrow \text{shorter seqs}, \, \text{larger embedding matrix}, \, \text{rare tail harder to learn}.
\]

## Prerequisites

- Chapter 01 (probability on discrete token IDs).
- Basic Unicode awareness (code points vs graphemes); optional reading on UTF-8.

## Chapter placement

| Connects to | Why tokenization matters |
|-------------|---------------------------|
| 03 Embeddings | \(|\mathcal{V}|\times d\) table size and OOV handling |
| 06 Pretraining | effective **tokens per epoch** vs unique documents |
| 08 Inference | byte-level vs word-level changes **speed** and **stop sequences** |

## Troubleshooting token mismatch

| Symptom | Likely cause |
|---------|----------------|
| Identical strings → different token counts | Normalisation rules differ (NFC, spaces) |
| PPL not comparable across models | Different vocab / BPE vs SentencePiece |
| Long sequences on CJK / emoji-heavy text | Byte-BPE elongation |

## Comparison snapshot

| Algorithm | Vocab build | Runtime segment | Typical stacks |
|-----------|-------------|-----------------|----------------|
| BPE | Greedy merges by frequency | Deterministic given merges | GPT-* family |
| WordPiece | Likelihood-oriented merges | Greedy / beam variants | BERT / DistilBERT |
| Unigram LM | Probabilistic tokens + EM-ish loop | Viterbi best path | T5, LLaMA SentencePiece |

Tokenizers are **not** interchangeable: switching changes \(|\mathcal{V}|\), special tokens, and **byte** handling.

## Mini glossary

| Term | Meaning |
|------|---------|
| Merge table | Ordered list of pair merges encoding the tokenizer runtime |
| Fertility | Tokens per linguistic word (rough proxy for efficiency) |
| BOS / EOS | Boundary markers; required for packing batches safely |

---

_End of chapter overview._
