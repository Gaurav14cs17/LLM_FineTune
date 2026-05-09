# Understanding WordPiece Tokenization (BERT)

*Likelihood-driven merges, PMI-style scoring, continuation markers, and contrast with BPE*

---

**WordPiece** (Schuster & Nakajima, 2012; adopted in BERT) constructs a **subword vocabulary** by iterative merges chosen **not** purely by raw bigram frequency (as in classical BPE) but by a **statistical association** score related to **pointwise mutual information**. BERT marks word-internal continuations with the **`##`** prefix so the encoder can distinguish **word-starts** from **suffix fragments**. The resulting **open vocabulary** handles rare/OOV surface forms without brittle character-only models.

**Reading map:** association score → PMI link → BERT **`##`** → contrast with greedy BPE.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 BPE vs WordPiece — Decision Rules](#11-bpe-vs-wordpiece--decision-rules)
   - [1.2 Objectives in Plain Language](#12-objectives-in-plain-language)
2. [Likelihood Gain and the WordPiece Score](#2-likelihood-gain-and-the-wordpiece-score)
3. [Connection to Pointwise Mutual Information](#3-connection-to-pointwise-mutual-information)
4. [Training Algorithm (Sketch)](#4-training-algorithm-sketch)
   - [4.1 Symbol Space and Merge Queue](#41-symbol-space-and-merge-queue)
   - [4.2 Stopping Criteria](#42-stopping-criteria)
   - [4.3 Corpus Normalisation and “Word” Boundaries](#43-corpus-normalisation-and-word-boundaries)
   - [4.4 Encoding New Text: Greedy Longest-Match-First](#44-encoding-new-text-greedy-longest-match-first)
   - [4.5 Smoothing Rare Pairs (Practical Stabiliser)](#45-smoothing-rare-pairs-practical-stabiliser)
5. [Worked Micro-Example](#5-worked-micro-example)
   - [5.1 Second Toy: Marginals Dominate Rare Bigram Frequency](#51-second-toy-marginals-dominate-rare-bigram-frequency)
6. [BERT Conventions: `##`, Special Tokens, Vocabulary Size](#6-bert-conventions--special-tokens-vocabulary-size)
7. [Comparison Table — BPE vs WordPiece](#7-comparison-table--bpe-vs-wordpiece)
8. [Pseudocode](#8-pseudocode)
9. [Common Mistakes](#9-common-mistakes)
10. [Exercises](#10-exercises)

---

## 1. Overview

### VARIABLES

| Symbol | Meaning |
|--------|---------|
| \(\mathcal{C}\) | Normalised training corpus of “words” after initial pre-tokenisation |
| \(f(\cdot)\) | Frequency counts of symbols or symbol pairs in the **current** segmentation |
| \(x,y\) | Adjacent symbols (characters or merged pieces) in some iteration |
| `score` | WordPiece merge criterion (see [Section 2](#2-likelihood-gain-and-the-wordpiece-score)) |
| \(|V|\) | Target vocabulary size (BERT-base \(\approx 30{,}522\) incl. specials) |
| `##` | **Continuation** marker in BERT WordPiece encoding |

### ┌─────────────────────────────────────────────────────────────┐
### │ INTUITION: glue only if pieces “stick” together            │
### └─────────────────────────────────────────────────────────────┘

```
BPE:  “Whoever types aa most wins today’s merge lottery.”
      → may explode frequent doubles before meaningful morphemes.

WP:   “Boost pair (x,y) if xy appears MORE than random independence
       predicts from marginals of x and y.”
      → merges **statistically glued** pieces, not merely frequent ones.

Example mental picture:
  freqs:  aa aa aa  (lots!)     but  each 'a' also appears everywhere
          ab       (moderate)   yet  'a' next to 'b' is *surprising* vs base rates
  → WordPiece may pick **ab** before **aa** if independence grossly underpredicts **ab**.
```

### 1.1 BPE vs WordPiece — Decision Rules

| Step driver | **BPE (GPT-style)** | **WordPiece (BERT-style score)** |
|-------------|--------------------|----------------------------------|
| Primary signal | **Count**\((x,y)\) | **Normalised** association \(\dfrac{f(xy)}{f(x)\,f(y)}\) |
| Philosophy | Compression via frequent pairs | Statistical cohesion of the pair |
| Ties broken | Usually higher raw count / lex order | Implementation-defined |

Both are **greedy** iterative procedures over a symbol inventory; neither solves **global** optimisation over all possible subword sets (contrast **Unigram LM** with EM pruning).

### 1.2 Objectives in Plain Language

**Informal likelihood story:** imagine a broken tokenizer emits tokens with empirical unigram probabilities \(\hat P(\cdot)\). Replacing explicit bigram emissions \(x\) then \(y\) by a merged symbol \(xy\) **raises** corpus probability when the **product** \(\hat P(x)\hat P(y)\) **underestimates** the empirical joint mass of typing \(xy\) as an **atomic** unit. The WordPiece score approximates that intuition.

---

## 2. Likelihood Gain and the WordPiece Score

Let **\(f(xy)\)** be the count of adjacent pair \(x y\) across current split tokens, **\(f(x)\)** marginal counting rules per the algorithm’s bookkeeping (often: symbol occurrences under the current segmentation; implementation details vary by toolkit).

**WordPiece pair score** (standard reference form):

\[
\boxed{\mathrm{score}(x,y) = \frac{f(xy)}{f(x)\,f(y)}.}
\]

**Interpretation:** if \(x\) and \(y\) were **independent draws** under independence-inspired marginals placed in the denominator, the ratio compares **observed adjacency mass** to a **product-of-marginals** baseline. Larger scores → stronger evidence that **\(xy\)** behaves like a unit.

---

## 3. Connection to Pointwise Mutual Information

For empirical distributions \(\hat p\) over symbol *pairs* and *singletons* in a (local) unigram + independent hypothesis,

\[
\mathrm{PMI}(x,y) = \log \frac{\hat p(x,y)}{\hat p(x)\,\hat p(y)}.
\]

If we identify \(\hat p(x,y) \propto f(xy)\), \(\hat p(x)\propto f(x)\), \(\hat p(y)\propto f(y)\), then

\[
\mathrm{score}(x,y) \;\propto\; \frac{\hat p(x,y)}{\hat p(x)\,\hat p(y)} \;=\; \exp\bigl(\mathrm{PMI}(x,y)\bigr).
\]

Thus high scores coincide with **positive PMI** — symbols co-occur adjacently **more** than independence suggests. **Log‑score** \(\log \mathrm{score}(x,y)\) aligns additively with PMI up to constants from **total token mass** bookkeeping.

---

## 4. Training Algorithm (Sketch)

### 4.1 Symbol Space and Merge Queue

1. **Initialise** vocabulary with **atomic** symbols (characters + maybe user seeds). Segment corpus into those atoms (BERT: often **whitespace-split** words → chars inside).
2. **Repeat** until \(|V|\) reaches target:
   - Enumerate **adjacent** candidate pairs in the current tokenisation.
   - Compute \(\mathrm{score}(x,y)\) from current counts \(f\).
   - **Pick** pair with **maximum** score (subject to thresholds/min frequency constraints in production cleaners).
   - **Merge** symbols into a new token `xy`, add to \(V\), **re-tokenise** corpus or update counts incrementally (**implementation-dependent**).
3. **Output** merge list + final vocabulary IDs.

### 4.2 Stopping Criteria

Stop when **size budget** \(|V|\) is met or gains fall below an \(\epsilon\) threshold. Production systems often **filter** merges that hurt **dev** perplexity on a held-out shard.

### 4.3 Corpus Normalisation and “Word” Boundaries

Modern trainers lowercase / strip accents **before** statistics (BERT **uncased** vs **cased** vocabs). **Word boundaries** matter: you typically accumulate pair statistics **inside** WordPiece’s pre-segmented units so that cross-word bigrams are **not** merged into a single piece accidentally (merging across spaces would entangle unrelated tokens).

### 4.4 Encoding New Text: Greedy Longest-Match-First

After training, **encoding** walks each word’s character sequence and **greedily** consumes the **longest** vocabulary prefix—a **trie** lookup yields \(O(\text{length})\) time. When **no** prefix matches, fall back to **single characters** until progress resumes. An **unknown-token** bucket surfaces tokens that cannot be encoded even at the character level (rare if **full** character coverage exists).

**Contrast with training:** training optimises **which merges enter** \(V\); inference **never** reconsider merges—it only **lookup**s.

### 4.5 Smoothing Rare Pairs (Practical Stabiliser)

Define smoothed marginals \(\tilde f(x) = f(x) + \alpha\) for small \(\alpha>0\) (symmetric Dirichlet prior intuition). Then

\[
\mathrm{score}_\alpha(x,y) = \frac{f(xy)}{\tilde f(x)\,\tilde f(y)}.
\]

As \(f(x)\to 0\), raw denominators explode; smoothing caps **wild** ratios when counts are tiny (useful on **skewed** web corpora).

---

## 5. Worked Micro-Example

**Corpus (tokenised words, counts indicated):**

- `"aaab"` appears **5** times.
- `"aaac"` appears **2** times.
- Assume **space-separated** input already yields these **word tokens** (toy setting avoids space fragmentation quirks).

Initial inside-word characters:
- In `"aaab"`: pairs \((a,a)\) occur **twice** per word (positions 0-1 and 1-2); \((a,b)\) once.
- Five such words ⇒ \(f(a,a)=10\) at substring level? **Careful:** per word of length 4, **overlapping** bigrams of characters: \((a_0,a_1),(a_1,a_2),(a_2,b)\). Count **pair** \((a,a)\): **2** per `"aaab"` ×5 = **10**; pair \((a,b)\): **5**.

In `"aaac"` ×2: \((a,a)\): **2** per word ×2 = **4**; total \(f(a,a)=14\); \(f(a,b)=5\); \(f(a,c)=2\) (last pair).

**Marginal character counts** \(f(a)\) count **character occurrences**:
- From `"aaab"`: **3** `a` per word ×5 = **15** `a`; from `"aaac"` ×2: **3**×2 = **6** `a` ⇒ \(f(a)=21\).
- \(f(b)=5\), \(f(c)=2\).

**Scores:**

\[
\mathrm{score}(a,a)=\frac{f(aa)}{f(a)\,f(a)}=\frac{14}{21\cdot 21}=\frac{14}{441}\approx 0.0317.
\]

\[
\mathrm{score}(a,b)=\frac{5}{21\cdot 5}=\frac{1}{21}\approx 0.0476.
\]

\[
\mathrm{score}(a,c)=\frac{2}{21\cdot 2}=\frac{1}{21}\approx 0.0476.
\]

**Result:** Pure **BPE** frequency might prioritise `(a,a)` (higher raw bigram count), but the **WordPiece** score lifts \((a,b)\) / \((a,c)\) above \((a,a)\) in this configuration—**meaningful glue** vs **repeated duplicates of `a`**.

**Takeaway:** WordPiece prefers pairs whose **joint** mass clears a **higher bar** relative to independent marginals.

### 5.1 Second Toy: Marginals Dominate Rare Bigram Frequency

Suppose \(f(xy)=100\) but \(f(x)=10{,}000\) and \(f(y)=10{,}000\) (both **super common** characters). Then

\[
\mathrm{score}(x,y)=\frac{100}{10^8}=10^{-6}.
\]

Another pair \((u,v)\) with \(f(uv)=20\), \(f(u)=f(v)=30\) yields

\[
\mathrm{score}(u,v)=\frac{20}{900}\approx 0.022 \gg 10^{-6}.
\]

Even though \(f(xy)\) dwarfs \(f(uv)\) in absolute terms, **normalisation by products of marginals** can **invert** merge ordering—this is the **core** departure from raw BPE frequency.

---

## 6. BERT Conventions: `##`, Special Tokens, Vocabulary Size

**Continuation marker `##`:**

```
Surface:       unhappiness
WordPiece:     un  ##happy  ##ness
               ↑            ↑
           word start    continuation fragments
```

The first subword token **lacks** `##`; subsequent pieces **include** it—BERT’s cue that they **continue** the same lexical word.

**Special tokens (conceptual):**

- `[CLS]`, `[SEP]`, `[MASK]`, `[unusedX]` slots in released vocabs.
- `<unk>`, `[PAD]` and similar entries map **unknown** spans and **padding** in batches.

**`[CLS]` / `[SEP]` placement** is a modelling contract: WordPiece alone does not insert them—**downstream featurisers** do.

**Vocabulary sizes:**

- BERT-base WordPiece \(\approx 30{,}522\) entries (with **reserved** special slots).
- Training corpora historically **English Wikipedia + BookCorpus** (now common to replicate procedure on other domains).

---

## 7. Comparison Table — BPE vs WordPiece

| Aspect | BPE | WordPiece (BERT) |
|--------|-----|------------------|
| Merge driver | Raw pair frequency | \(\dfrac{f(xy)}{f(x)f(y)}\) |
| Typical merges early | Very frequent repeated letters | Statistically sticky pairs |
| Continuation marking | GPT often uses `Ġ` / space-byte | `##` continuation |
| Greedy? | yes | yes |
| Global likelihood | **No** closed-form optimum | **No** |

Both can achieve **open-vocabulary** coverage when backed by a **character/base set**.

---

## 8. Pseudocode

```
# counts: maintain corpus-wide statistics under CURRENT segmentation

def wordpiece_pair_score(pair_xy, f_marginal, f_pair):
    x, y = pair_xy
    return f_pair[pair_xy] / (f_marginal[x] * f_marginal[y])

def train_wordpiece(corpus_tokens, target_V, seed_vocab):
    V = copy(seed_vocab)
    tokenisation = [list(w) for w in corpus_tokens]  # toy char split
    while len(V) < target_V:
        pair_counts = count_adjacent_pairs(tokenisation)
        marginal_counts = count_symbols(tokenisation)
        best_pair = None
        best_score = -inf
        for (x,y), fxy in pair_counts.items():
            s = wordpiece_pair_score((x,y), marginal_counts, {(x,y): fxy})
            if s > best_score:
                best_score, best_pair = s, (x,y)
        if best_pair is None:
            break
        new_symbol = merge_symbols(best_pair)
        add new_symbol to V
        tokenisation = apply_merge_everywhere(tokenisation, best_pair, new_symbol)
    return V

# encoding uses LONGEST-match or trie over V with ## rules in BERT tooling
```

**Encoder-side** logic in transformers’ `WordpieceTokenizer` is **deterministic** given a fixed vocabulary: greedy **left-to-right maximal** matches with unknown fallbacks.

---

## 9. Common Mistakes

- ❌ **Identical denominators across toolkits** — \(f(x)\) definitions differ (count **tokens** vs **positions**). ✓ **Always** reproduce scores with **one** counting convention.
- ❌ **Thinking `##` influences training merges** — marker is a **surface convention** for **encoding**, not the merge objective. ✓ Strip / annotate **after** piece identities exist.
- ❌ **Equating high `f(xy)` with high `score`** — huge marginals \(f(x)f(y)\) **penalise** overly common atoms. ✓ **Normalise** via the product.
- ❌ **Assuming WordPiece is “less greedy” than BPE globally** — both are **sequence of myopic merges**. ✓ For global likelihood under a **unigram lexicon**, see **SentencePiece Unigram LM**.

---

## 10. Exercises

1. **Recompute** \(\mathrm{score}(a,a)\), \(\mathrm{score}(a,b)\) if you **double** the corpus size uniformly (all counts ×2). Prove the **ratio** of scores is **unchanged** if all numerators/denominators scale homogeneously — when does scaling **not** commute?
2. **Algebra:** with \(\hat p(x,y)=f(xy)/N\), \(\hat p(x)=f(x)/N\), show \(\log \mathrm{score}(x,y) = \mathrm{PMI}(x,y) + \log N\) under this simple event space (track **what \(N\) counts**).
3. **BERT detail:** explain why `##` helps **restore** word boundaries when **subword** pieces are fed to the model—what would break if continuation lacked a marker?
4. **Design:** propose a **tie-break** policy when two different pairs share the **same** score—discuss stability across runs and **determinism** in distributed training.

---

*WordPiece injects a **normalised cohesion** signal into greedy merging—distinct from raw BPE frequency, yet still only a **local** heuristic compared to full **subword language-model** pruning.*
