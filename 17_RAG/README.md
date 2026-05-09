# Understanding RAG: Retrieval-Augmented Generation

*From vector search to chunking strategies — building knowledge-grounded LLM systems*

---

**RAG (Retrieval-Augmented Generation)** grounds LLM outputs in external, up-to-date knowledge. Instead of relying solely on memorised training data, RAG retrieves relevant documents at inference time and injects them into the LLM's context window. This eliminates hallucination for factual queries and enables domain-specific answering without fine-tuning.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why RAG?](#11-why-rag)
   - [1.2 RAG vs Fine-Tuning](#12-rag-vs-fine-tuning)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Document Processing](#2-document-processing)
   - [2.1 Chunking Strategies](#21-chunking-strategies)
   - [2.2 Overlap and Context Windows](#22-overlap-and-context-windows)
3. [Embedding and Indexing](#3-embedding-and-indexing)
   - [3.1 Dense Retrieval (Bi-Encoder)](#31-dense-retrieval-bi-encoder)
   - [3.2 Approximate Nearest Neighbours (ANN)](#32-approximate-nearest-neighbours-ann)
   - [3.3 Cosine Similarity vs Dot Product](#33-cosine-similarity-vs-dot-product)
4. [Retrieval](#4-retrieval)
   - [4.1 Top-K Retrieval](#41-top-k-retrieval)
   - [4.2 Hybrid Search (Dense + Sparse)](#42-hybrid-search-dense--sparse)
   - [4.3 Reranking](#43-reranking)
5. [Generation with Context](#5-generation-with-context)
   - [5.1 Prompt Construction](#51-prompt-construction)
   - [5.2 Lost-in-the-Middle Problem](#52-lost-in-the-middle-problem)
6. [Evaluation Metrics](#6-evaluation-metrics)
   - [6.1 Retrieval Metrics](#61-retrieval-metrics)
   - [6.2 Generation Quality](#62-generation-quality)
7. [Summary](#7-summary)
   - [7.1 Formulas Quick Reference](#71-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 Why RAG?

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: LLMs have STALE and UNRELIABLE knowledge          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM alone:                                                 │
│  Q: "What is the latest version of PyTorch?"                │
│  A: "PyTorch 2.0" ← WRONG (training data cutoff!)          │
│                                                             │
│  LLM + RAG:                                                 │
│  Step 1: Search docs → finds "PyTorch 2.6 released 2025"   │
│  Step 2: Inject into prompt                                 │
│  A: "PyTorch 2.6" ← CORRECT (grounded in retrieved doc!)   │
│                                                             │
│  THREE BENEFITS:                                            │
│  1. FRESHNESS: access up-to-date information               │
│  2. VERIFIABILITY: cite sources, check facts               │
│  3. DOMAIN SPECIFICITY: use your company's private docs     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge update | Instant (update docs) | Requires retraining |
| Cost | Low (no training) | High (GPU hours) |
| Hallucination | Low (grounded in docs) | Can still hallucinate |
| Domain adaptation | Good for factual QA | Better for style/behaviour |
| Latency | Higher (retrieval + generation) | Lower (generation only) |
| Context length needed | Yes (inject retrieved docs) | No |

### 1.3 Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│  OFFLINE (indexing):                                        │
│  Documents → Chunk → Embed → Store in Vector DB            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ONLINE (query time):                                       │
│                                                             │
│  User query                                                 │
│      ↓                                                      │
│  1. EMBED query → q ∈ ℝᵈ                                    │
│      ↓                                                      │
│  2. RETRIEVE top-K chunks from vector DB                    │
│     (cos_sim(q, doc_i) → rank → select top-K)             │
│      ↓                                                      │
│  3. (Optional) RERANK with cross-encoder                    │
│      ↓                                                      │
│  4. CONSTRUCT prompt: [system] + [retrieved docs] + [query] │
│      ↓                                                      │
│  5. GENERATE answer with LLM (grounded in docs)             │
│      ↓                                                      │
│  Output: answer + source citations                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Document Processing

### 2.1 Chunking Strategies

```
STRATEGY 1: FIXED-SIZE CHUNKS (simplest)
  Split every N tokens (e.g., N=512)
  + Simple, predictable
  - May split mid-sentence or mid-paragraph

STRATEGY 2: RECURSIVE/SEMANTIC SPLITTING
  Split on natural boundaries: paragraphs > sentences > tokens
  + Preserves meaning within chunks
  - Variable chunk sizes

STRATEGY 3: SLIDING WINDOW
  Chunks of size N with overlap O (e.g., N=512, O=100)
  + No information lost at boundaries
  - More chunks to index (higher storage cost)

OPTIMAL CHUNK SIZE TRADE-OFF:
  ┌─────────────────────────────────────────────────────────────┐
  │  Small chunks (128 tokens):                                 │
  │    + Precise retrieval (less noise in each chunk)           │
  │    - Less context per chunk (may miss relevant info nearby) │
  │    - Many chunks to search through                          │
  │                                                             │
  │  Large chunks (2048 tokens):                                │
  │    + More context per retrieval                             │
  │    - More noise (irrelevant text mixed in)                  │
  │    - Fewer chunks fit in context window                     │
  │                                                             │
  │  Sweet spot: 256–512 tokens with 50-100 token overlap      │
  └─────────────────────────────────────────────────────────────┘
```

### 2.2 Overlap and Context Windows

```
EXAMPLE: document of 1500 tokens, chunk_size=512, overlap=100

  Chunk 1: tokens [0, 512)      ← 512 tokens
  Chunk 2: tokens [412, 924)    ← overlap of 100 with Chunk 1
  Chunk 3: tokens [824, 1336)   ← overlap of 100 with Chunk 2
  Chunk 4: tokens [1236, 1500)  ← final partial chunk (264 tokens)

  Number of chunks: ⌈(1500 - 512) / (512 - 100)⌉ + 1 = ⌈988/412⌉ + 1 = 4

  WHY OVERLAP?
  Without overlap: "The XYZ method (described in previous section) achieves 95%"
    → "described in previous section" refers to text in a DIFFERENT chunk!
    → Retrieval misses the connection.
  
  With overlap: both chunks contain the boundary text → context preserved.
```

---

## 3. Embedding and Indexing

### 3.1 Dense Retrieval (Bi-Encoder)

```
BI-ENCODER ARCHITECTURE:
  Query encoder:    q = E_q(text_query) ∈ ℝᵈ     (d=768 or 1024)
  Document encoder: d = E_d(text_chunk) ∈ ℝᵈ

  Score: sim(q, d) = cos(q, d) = q·d / (‖q‖×‖d‖)

  TRAINING: contrastive loss (InfoNCE)
    L = -log(exp(sim(q, d⁺)/τ) / Σᵢ exp(sim(q, dᵢ)/τ))
    
    d⁺ = positive document (relevant to query)
    dᵢ = all documents in batch (positives + in-batch negatives)
    τ = temperature (0.01-0.1)

POPULAR EMBEDDING MODELS:
  ┌───────────────┬─────┬────────────────────────────────┐
  │ Model         │ dim │ Notes                          │
  ├───────────────┼─────┼────────────────────────────────┤
  │ E5-large-v2   │ 1024│ Strong general-purpose         │
  │ GTE-large     │ 1024│ Alibaba, multilingual          │
  │ bge-large-en  │ 1024│ BAAI, fine-tuned for retrieval │
  │ text-emb-3-sm │ 1536│ OpenAI, API-based              │
  │ nomic-embed   │ 768 │ Open source, efficient         │
  └───────────────┴─────┴────────────────────────────────┘
```

### 3.2 Approximate Nearest Neighbours (ANN)

```
PROBLEM: exact kNN on 10M vectors × 1024 dims = very slow!
  Brute force: O(N × d) per query = 10M × 1024 = 10B ops

ANN SOLUTIONS:
  HNSW (Hierarchical Navigable Small World):
    Build a multi-layer graph where each node = one embedding.
    Search: start at top layer, greedily traverse to closest node,
    then descend to next layer and repeat.
    
    Query time: O(log N × d)   (typically 1-5 ms for 10M vectors)
    Memory:     O(N × d + N × M)  where M = graph connections
    Recall@10:  95-99% of exact kNN

  IVF (Inverted File):
    Cluster vectors into C centroids.
    Query: find closest nprobe centroids, search only those clusters.
    
    Query time: O(nprobe × N/C × d)
    Good for: very large collections (100M+)

PRACTICAL CHOICES:
  < 1M vectors:    HNSW (high recall, low latency)
  1M-100M:         IVF-PQ (compressed vectors, larger scale)
  > 100M:          distributed index (Pinecone, Weaviate)
```

### 3.3 Cosine Similarity vs Dot Product

```
COSINE SIMILARITY:
  cos(q, d) = q·d / (‖q‖ × ‖d‖)   ∈ [-1, 1]
  
  Invariant to vector magnitude — only considers DIRECTION.
  Good when embeddings have varying norms.

DOT PRODUCT:
  sim(q, d) = q·d   ∈ (-∞, ∞)
  
  Considers both direction AND magnitude.
  If embeddings are L2-normalised: dot product = cosine similarity.

MAXIMUM INNER PRODUCT SEARCH (MIPS):
  When using dot product with non-normalised embeddings,
  need MIPS-specific algorithms (not standard kNN).
  
  Solution: L2-normalise all embeddings → cos ≡ dot product → use kNN.
  This is what most production systems do.
```

---

## 4. Retrieval

### 4.1 Top-K Retrieval

```
Given query embedding q:
  1. Compute similarity: sᵢ = cos(q, dᵢ) for all documents dᵢ
  2. Sort by similarity: top-K documents with highest sᵢ
  3. Return: [(doc_1, s_1), (doc_2, s_2), ..., (doc_K, s_K)]

CHOOSING K:
  K=3:   minimal context, fast, might miss relevant info
  K=5:   good balance for most tasks
  K=10:  more context, higher recall, risks including irrelevant docs
  K=20:  use with reranking (retrieve many, rerank to select best)

CONTEXT BUDGET:
  Context window: 8192 tokens
  System prompt:  ~200 tokens
  User query:     ~50 tokens
  Retrieved docs: K × chunk_size ≤ 8192 - 200 - 50 - generation_budget
  
  For K=5, chunk_size=512: 2560 tokens used for retrieval (31% of context)
```

### 4.2 Hybrid Search (Dense + Sparse)

```
DENSE RETRIEVAL (embeddings):
  + Good for semantic similarity ("synonyms", "paraphrases")
  + Handles meaning even when words differ
  - Bad for exact keyword matching (rare names, IDs, codes)
  
SPARSE RETRIEVAL (BM25/TF-IDF):
  + Excellent for exact terms, proper nouns, codes
  + No training needed, fast
  - Bad for semantic understanding (different words = no match)

HYBRID = BEST OF BOTH:
  score_hybrid = α × score_dense + (1-α) × score_sparse
  
  Typical α = 0.7 (weight dense higher, but keep sparse for exact match)
  
  RECIPROCAL RANK FUSION (alternative to weighted sum):
    RRF(d) = Σ 1/(k + rank_system(d))    for each retrieval system
    k = 60 (constant to prevent over-weighting top results)
```

### 4.3 Reranking

```
CROSS-ENCODER RERANKER:
  Unlike bi-encoder (separate query and doc encoding):
  Cross-encoder processes [query, document] TOGETHER.
  
  score = CrossEncoder([CLS] query [SEP] document [SEP])
  
  Much more accurate (sees full interaction between q and d)
  But: O(N × T) inference cost — too expensive for full corpus!
  
  SOLUTION: two-stage retrieval
  Stage 1: Bi-encoder retrieves top-100 (fast, O(1) per doc)
  Stage 2: Cross-encoder reranks top-100 to select top-5 (slow but accurate)
  
  COMMON RERANKERS:
  - BGE-reranker-v2-m3
  - Cohere rerank
  - jina-reranker-v2
```

---

## 5. Generation with Context

### 5.1 Prompt Construction

```
TEMPLATE:
  ┌────────────────────────────────────────────────────────────┐
  │ SYSTEM: You are a helpful assistant. Answer the question   │
  │ based ONLY on the provided context. If the answer is not   │
  │ in the context, say "I don't have enough information."     │
  │                                                            │
  │ CONTEXT:                                                   │
  │ [Document 1] (source: doc_name_1.pdf, page 3)             │
  │ The transformer architecture was introduced in 2017...     │
  │                                                            │
  │ [Document 2] (source: doc_name_2.pdf, page 12)            │
  │ Attention mechanism allows tokens to interact...           │
  │                                                            │
  │ [Document 3] (source: doc_name_3.pdf, page 1)             │
  │ Self-attention has O(n²) complexity...                     │
  │                                                            │
  │ QUESTION: {user_query}                                     │
  └────────────────────────────────────────────────────────────┘
```

### 5.2 Lost-in-the-Middle Problem

```
┌─────────────────────────────────────────────────────────────┐
│  OBSERVATION (Liu et al. 2023):                             │
│  LLMs attend less to documents placed in the MIDDLE         │
│  of the context window.                                     │
│                                                             │
│  Attention weight by document position:                     │
│  Position 1 (beginning):  HIGH attention  ← model reads     │
│  Position 2:              medium                            │
│  Position 3 (middle):     LOW attention   ← "lost!"        │
│  Position 4:              medium                            │
│  Position 5 (end):        HIGH attention  ← model reads     │
│                                                             │
│  SOLUTIONS:                                                 │
│  1. Place most relevant doc FIRST (highest similarity)      │
│  2. Place second-most-relevant doc LAST                     │
│  3. Keep total K small (≤5 documents)                        │
│  4. Summarise retrieved docs before injecting              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Evaluation Metrics

### 6.1 Retrieval Metrics

```
RECALL@K: fraction of relevant docs in top-K retrieval
  Recall@K = |{relevant} ∩ top-K| / |{relevant}|

  Recall@5 = 0.8 means: 80% of relevant docs appear in top 5

NDCG@K (Normalised Discounted Cumulative Gain):
  Measures not just WHAT was retrieved, but IN WHAT ORDER.
  
  DCG@K = Σᵢ₌₁ᴷ (2^{rel_i} - 1) / log₂(i+1)
  NDCG@K = DCG@K / IDCG@K   (normalised by ideal ordering)
  
  Penalises relevant docs ranked low.

MRR (Mean Reciprocal Rank):
  For first relevant result at rank r: RR = 1/r
  MRR = average of 1/r across queries
  
  MRR = 1.0: every query's answer is rank 1 (perfect)
  MRR = 0.5: on average, first answer at rank 2
```

### 6.2 Generation Quality

```
FAITHFULNESS: does the answer use ONLY the retrieved context?
  Metric: check if claims in answer are supported by retrieved docs.
  Common: LLM-as-judge ("Is claim X supported by context Y?")

RELEVANCE: does the answer actually address the question?
  Metric: semantic similarity between answer and question intent.

GROUNDEDNESS: are all statements traceable to a source?
  Metric: fraction of answer sentences with a supporting document.
```

---

## 7. Summary

### 7.1 Formulas Quick Reference

**Cosine similarity:**

```
cos(q, d) = q·d / (‖q‖ × ‖d‖)
```

**InfoNCE loss (embedding training):**

```
L = -log(exp(sim(q,d⁺)/τ) / Σᵢ exp(sim(q,dᵢ)/τ))
```

**Hybrid scoring:**

```
score = α × norm_dense + (1-α) × norm_sparse   (α ≈ 0.7)
```

**Chunk count:**

```
N_chunks = ⌈(doc_length - chunk_size) / (chunk_size - overlap)⌉ + 1
```

| Component | Latency | Comment |
|-----------|---------|---------|
| Query embedding | 5-10 ms | One forward pass through E_q |
| ANN search (HNSW) | 1-5 ms | Scales logarithmically with N |
| Reranking (top-20) | 50-200 ms | Cross-encoder on 20 pairs |
| LLM generation | 500-3000 ms | Depends on output length |
| **Total** | **~600-3200 ms** | Dominated by LLM generation |

### 7.2 Common Mistakes

```
❌ WRONG: Bigger chunks are always better (more context per retrieval)
✓ RIGHT:  Bigger chunks dilute the relevant information with noise.
          The embedding represents the AVERAGE semantics of the chunk.
          A 2000-token chunk about 5 topics → embedding is vague about all 5.
          Better: 400-token focused chunks + metadata for parent document.

❌ WRONG: Just throw all retrieved docs into the context
✓ RIGHT:  Order matters (lost-in-the-middle), quantity matters (too many
          = noise overwhelms signal), and relevance threshold matters.
          Filter: only include docs with similarity > threshold (e.g., 0.7).

❌ WRONG: Fine-tuning the LLM on your docs is a substitute for RAG
✓ RIGHT:  Fine-tuning on domain docs helps the model's style and vocabulary,
          but it CANNOT memorise all facts reliably. RAG provides a verifiable
          source of truth. Best approach: RAG + fine-tuned model.

❌ WRONG: RAG eliminates all hallucination
✓ RIGHT:  RAG reduces hallucination significantly but doesn't eliminate it.
          The model can still hallucinate if:
          (a) No relevant doc is retrieved (retrieval failure)
          (b) The model ignores the context (instruction-following failure)
          (c) The retrieved doc itself contains errors (data quality)
```

---

## 8. Exercises

1. **Chunking**: A 10,000-token document is chunked with size=512 and overlap=100. How many chunks are created? What is the total token count across all chunks (accounting for overlaps)?

2. **Retrieval Math**: Given query embedding q=[0.5, 0.5, 0.5] (normalised: ‖q‖=0.866) and 3 document embeddings: d₁=[1,0,0], d₂=[0.4,0.4,0.4], d₃=[0,0,1]. Compute cosine similarity for each. Which is retrieved as top-1?

3. **Hybrid Scoring**: For a query, dense retrieval returns [doc_A:0.9, doc_B:0.7, doc_C:0.5] and BM25 returns [doc_C:0.8, doc_A:0.6, doc_B:0.3]. With α=0.7, compute hybrid scores and final ranking.

4. **Context Budget**: Context window = 4096 tokens. System prompt = 150 tokens, user query = 80 tokens, max generation = 500 tokens. How many tokens remain for retrieved context? With chunk_size=400, what is the maximum K?
