# Understanding Large Language Models: A Mathematical Deep Dive

*A comprehensive, chapter-by-chapter guide to LLMs — from probability basics to scaling laws*

---

This repository is a self-contained **BookWise chapter** on Large Language Models. Every topic is presented in the style of a rigorous tutorial: intuition first, formal mathematics second, concrete numerical examples third, and working pseudocode throughout.

The content covers everything needed to understand, train, and deploy a modern LLM like LLaMA-3 8B.

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
   - [1.1 What This Covers](#11-what-this-covers)
   - [1.2 How to Navigate](#12-how-to-navigate)
   - [1.3 Prerequisites](#13-prerequisites)
2. [Chapter Map](#2-chapter-map)
   - [2.1 Learning Path](#21-learning-path)
   - [2.2 Quick Reference by Topic](#22-quick-reference-by-topic)
3. [Key Numbers to Remember](#3-key-numbers-to-remember)
   - [3.1 LLaMA-3 8B Architecture](#31-llama-3-8b-architecture)
   - [3.2 Training & Inference Numbers](#32-training--inference-numbers)
4. [Mathematical Conventions](#4-mathematical-conventions)
5. [Common Mistakes FAQ](#5-common-mistakes-faq)
6. [References](#6-references)

---

## 1. Repository Overview

### 1.1 What This Covers

**Large Language Models** are probabilistic text generators trained on massive corpora. At their core they are:

```
┌─────────────────────────────────────────────────────────────┐
│  WHAT AN LLM IS:                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  A neural network that estimates:                           │
│    P(next token | all previous tokens)                      │
│                                                             │
│  Architecture: Transformer (attention + FFN) stacked L times│
│  Size:         4B – 700B+ parameters                        │
│  Training:     Minimise NLL = −(1/T) Σ log P(wₜ|w<t)       │
│  Generation:   Autoregressive sampling from P(·|context)    │
│                                                             │
│  This repo covers every component of that pipeline,        │
│  with full mathematical derivations.                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 How to Navigate

```
READING PATHS:

  Linear (recommended for newcomers):
    Ch 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

  Just the architecture:
    Ch 3 (Embeddings) → 4 (Attention) → 5 (Transformer)

  Just fine-tuning:
    Ch 1 (Probability basics) → 7 (Fine-Tuning)

  Just inference:
    Ch 4 (Attention) → 8 (Inference)

  Just scaling:
    Ch 6 (Pre-training) → 9 (Scaling Laws)
```

Each sub-chapter `README.md` follows this structure:
1. **Intuition box** — real-world analogy and ASCII diagrams
2. **Formal mathematics** — full derivations with every step shown
3. **Numerical examples** — concrete numbers, not just symbols
4. **Pseudocode** — implementation-ready algorithms
5. **Common mistakes** — what to watch out for
6. **Exercises** — test your understanding

### 1.3 Prerequisites

| Topic | Required? | Where to review |
|-------|-----------|----------------|
| Calculus (derivatives, chain rule) | ✅ Essential | Ch 1 covers what's needed |
| Linear algebra (matrix multiply) | ✅ Essential | All chapters use it |
| Probability basics | ✅ Essential | Ch 1 covers it |
| Python/NumPy | 🔧 Helpful | For pseudocode |
| Neural networks (basics) | 🔧 Helpful | Pre-transformer DL |
| PyTorch | ⚡ Optional | For implementation |

---

## 2. Chapter Map

### 2.1 Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 1: Foundations                                     │
│  P(w₁,...,wₙ) = ∏ P(wₜ|w<t)    ← the LLM identity         │
│  Cross-entropy, KL divergence, perplexity                   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 2: Tokenization                                    │
│  Raw text → integer token IDs                               │
│  BPE, WordPiece, Unigram LM                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 3: Embeddings                                      │
│  Token IDs → dense vectors                                  │
│  Sinusoidal PE, RoPE, ALiBi                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 4: Attention                                       │
│  Attention = softmax(QKᵀ/√d_k + M) × V                     │
│  MHA, GQA, FlashAttention                                   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 5: Transformer Architecture                        │
│  RMSNorm, SwiGLU, residual connections                      │
│  Full forward pass: token → logits                          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        ↓                                     ↓
┌───────────────────┐               ┌─────────────────────┐
│  CHAPTER 6:       │               │  CHAPTER 8:         │
│  Pre-training     │               │  Inference          │
│  AdamW, mixed     │               │  KV cache, sampling │
│  precision, FLOP  │               │  speculative decode │
│  estimates        │               └─────────────────────┘
└────────┬──────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 7: Fine-Tuning                                     │
│  SFT, LoRA, QLoRA, RLHF-PPO, DPO                           │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  CHAPTER 9: Scaling Laws                                    │
│  L(N,D) = E + A/N^α + B/D^β                                 │
│  Kaplan, Chinchilla, emergence                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Quick Reference by Topic

| You want to understand... | Go to |
|--------------------------|-------|
| Why `1/√d_k` in attention | Ch 4.01 §3.1 |
| How LoRA reduces memory | Ch 7.02 §6.1 |
| Why cosine LR schedule | Ch 6.01 §5 |
| What RoPE does | Ch 3.03 |
| How DPO replaces PPO | Ch 7.05 |
| KV cache memory formula | Ch 8.03 §3 |
| Chinchilla optimal recipe | Ch 9.02 §3 |
| Why BF16 training | Ch 6.03 |
| SwiGLU activation | Ch 5.02 |
| Speculative decoding | Ch 8.04 |

---

## 3. Key Numbers to Remember

### 3.1 LLaMA-3 8B Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  LLAMA-3 8B: ARCHITECTURE AT A GLANCE                        │
├────────────────────────────┬─────────────────────────────────┤
│  Parameter                 │  Value                          │
├────────────────────────────┼─────────────────────────────────┤
│  Vocabulary size |V|       │  128,256                        │
│  Context window            │  8,192 (extended to 128K)       │
│  Model dimension d         │  4,096                          │
│  Layers L                  │  32                             │
│  Attention heads h_q       │  32                             │
│  KV heads h_kv (GQA)       │  8                              │
│  Head dimension d_k        │  128                            │
│  FFN intermediate dim      │  14,336                         │
│  Activation                │  SwiGLU                         │
│  Normalization             │  RMSNorm (ε = 1×10⁻⁵)           │
│  Position encoding         │  RoPE (θ = 500,000)             │
│  Total parameters          │  8.03B                          │
└────────────────────────────┴─────────────────────────────────┘
```

### 3.2 Training & Inference Numbers

```
TRAINING:
  Data:           15 trillion tokens
  FLOPs:          C ≈ 6 × 8B × 15T = 7.2 × 10²³
  Optimizer:      AdamW (β₁=0.9, β₂=0.95, λ=0.1)
  LR schedule:    Cosine warmup (peak: 3×10⁻⁴, min: 3×10⁻⁵)
  Precision:      BF16 activations, FP32 master weights
  Memory (full):  ~128 GB for full FT on 8B

INFERENCE (batch=1, context=8192):
  FLOPs per token: ≈ 2 × 8B = 16G FLOPs
  KV cache size:  128 KB × 8192 = 1.05 GB (BF16, GQA)
  Throughput (A100): ~100-200 tokens/s (w/ vLLM)
  Latency:         5-10ms per token (single A100)
```

---

## 4. Mathematical Conventions

| Notation | Meaning |
|----------|---------|
| d | Model hidden dimension |
| d_k, d_v | Key/query, value dimension per head |
| h | Number of attention heads |
| L | Number of transformer layers |
| n | Sequence length (number of tokens) |
| T | Training steps |
| N | Number of model parameters |
| D | Number of training tokens |
| C | Compute budget (FLOPs) |
| W | Weight matrix |
| θ | All model parameters (vector) or rotation angle |
| ⊙ | Element-wise multiplication |
| ‖·‖ | L2 norm (unless subscript says otherwise) |
| σ | Sigmoid function: σ(x) = 1/(1+e^{-x}) |
| log | Natural logarithm (base e) unless stated otherwise |

---

## 5. Common Mistakes FAQ

```
Q: Why is attention scaled by 1/√d_k and not 1/d_k?
A: Because Var[q·k] = d_k, so std = √d_k.
   To achieve unit variance after scaling: divide by √d_k.

Q: Why use RMSNorm instead of LayerNorm?
A: RMSNorm omits mean subtraction (redundant given biases exist).
   It's ~25% faster and performs equally well in practice.

Q: Why does LLaMA use more tokens than Chinchilla-optimal?
A: Chinchilla optimises training FLOPs. LLaMA optimises inference cost.
   Overtrained small models are cheaper to serve at scale.

Q: What is the difference between Adam and AdamW?
A: Weight decay. Adam adds L2 penalty to the gradient (wrong — interacts
   with adaptive scaling). AdamW adds weight decay directly to weights
   (correct — independent of gradient scale).

Q: Can LoRA achieve the same quality as full fine-tuning?
A: Yes, with sufficient rank. Theoretically: if rank(ΔW*) ≤ r, LoRA
   exactly represents the optimal update. In practice, r=16-64 achieves
   95-99% of full FT quality with 0.1-1% of the parameters.

Q: Why doesn't the KV cache grow unboundedly?
A: It does grow linearly. For n tokens, KV cache = O(n × L × d).
   That's why batched inference uses PagedAttention to manage memory.
```

---

## 6. References

1. **Vaswani et al. (2017)**: "Attention Is All You Need" — original Transformer
2. **Brown et al. (2020)**: "Language Models are Few-Shot Learners" — GPT-3
3. **Kaplan et al. (2020)**: "Scaling Laws for Neural Language Models"
4. **Hoffmann et al. (2022)**: "Training Compute-Optimal LLMs" — Chinchilla
5. **Touvron et al. (2023)**: "LLaMA: Open and Efficient Foundation LMs"
6. **Hu et al. (2022)**: "LoRA: Low-Rank Adaptation of LLMs"
7. **Rafailov et al. (2023)**: "Direct Preference Optimization (DPO)"
8. **Su et al. (2022)**: "RoFormer: Enhanced Transformer with Rotary PE"
9. **Dao et al. (2022)**: "FlashAttention: Fast and Memory-Efficient Attention"
10. **Kwon et al. (2023)**: "Efficient Memory Management for LLM Serving — vLLM"

---

## Folder Structure

```
LLM_FineTune/
├── README.md                          ← This file
│
├── 01_Foundations/
│   ├── README.md                      ← Chapter overview
│   ├── 01_Probability_Basics/         ← Chain rule, Bayes, E[X], Var[X]
│   ├── 02_Language_Model_Definition/  ← NLL loss, perplexity
│   └── 03_Entropy_and_Information/    ← KL divergence, cross-entropy
│
├── 02_Tokenization/
│   ├── README.md
│   ├── 01_BPE/                        ← Byte-pair encoding algorithm
│   ├── 02_WordPiece/                  ← WordPiece (BERT-style)
│   └── 03_SentencePiece_Unigram/      ← Unigram LM tokenizer
│
├── 03_Embeddings/
│   ├── README.md
│   ├── 01_Token_Embeddings/           ← Lookup table, initialisation
│   ├── 02_Sinusoidal_PE/              ← Original Transformer PE
│   └── 03_RoPE/                       ← Rotary position embedding
│
├── 04_Attention/
│   ├── README.md
│   ├── 01_Scaled_Dot_Product/         ← SDPA, scaling proof, masking
│   ├── 02_Multi_Head_Attention/       ← MHA, GQA, MQA
│   └── 03_FlashAttention/             ← IO-aware, O(n) memory
│
├── 05_Transformer_Architecture/
│   ├── README.md
│   ├── 01_RMSNorm/                    ← Pre-norm, scale invariance
│   ├── 02_SwiGLU_FFN/                 ← Gated activation, width
│   └── 03_Residual_Connections/       ← Highway gradient paths
│
├── 06_Pretraining/
│   ├── README.md
│   ├── 01_AdamW/                      ← Bias correction, weight decay
│   ├── 02_LR_Schedules/               ← Cosine warmup, WSD
│   └── 03_Mixed_Precision/            ← BF16, FP8, loss scaling
│
├── 07_FineTuning/
│   ├── README.md
│   ├── 01_SFT/                        ← Supervised fine-tuning
│   ├── 02_LoRA/                       ← Low-rank adaptation, r, α
│   ├── 03_QLoRA/                      ← 4-bit base model + LoRA
│   ├── 04_RLHF_PPO/                   ← Bradley-Terry, PPO, GAE
│   └── 05_DPO/                        ← Z(x) cancellation proof
│
├── 08_Inference/
│   ├── README.md
│   ├── 01_Decoding_Strategies/        ← Greedy, beam, sampling
│   ├── 02_Speculative_Decoding/       ← Draft model speedup
│   └── 03_KV_Cache/                   ← Memory formula, GQA, paging
│
├── 09_Scaling_Laws/
│   ├── README.md
│   ├── 01_Kaplan_Power_Laws/          ← N/D/C power law fits
│   └── 02_Chinchilla/                 ← D/N≈20, derivation
│
└── canvases/
    └── llm-bookwise-chapter.canvas.tsx ← Interactive React canvas
```
