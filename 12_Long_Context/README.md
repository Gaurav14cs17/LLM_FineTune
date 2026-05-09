# Understanding Long-Context LLMs

*From 4K to 1M+ tokens — the mathematics of extending context windows*

---

**Long-context modelling** enables LLMs to process entire codebases, books, and long conversations in a single prompt. Extending from 4K (GPT-2) to 128K (LLaMA-3) and 1M+ (Gemini) tokens requires solving position extrapolation, O(n²) memory, and attention degradation.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Long Context Matters](#11-why-long-context-matters)
   - [1.2 The O(n²) Problem](#12-the-on-problem)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Position Extrapolation Methods](#2-position-extrapolation-methods)
   - [2.1 Position Interpolation (PI)](#21-position-interpolation-pi)
   - [2.2 NTK-Aware Scaling](#22-ntk-aware-scaling)
   - [2.3 YaRN (Yet Another RoPE Extension)](#23-yarn-yet-another-rope-extension)
3. [Efficient Attention for Long Sequences](#3-efficient-attention-for-long-sequences)
   - [3.1 Ring Attention](#31-ring-attention)
   - [3.2 Sparse Attention Patterns](#32-sparse-attention-patterns)
4. [Memory Optimizations](#4-memory-optimizations)
   - [4.1 Sliding Window Attention](#41-sliding-window-attention)
   - [4.2 Token Eviction and Compression](#42-token-eviction-and-compression)
5. [Training Long-Context Models](#5-training-long-context-models)
   - [5.1 Progressive Length Training](#51-progressive-length-training)
   - [5.2 Long-Context Data](#52-long-context-data)
6. [Summary](#6-summary)
   - [6.1 Formulas Quick Reference](#61-formulas-quick-reference)
   - [6.2 Common Mistakes](#62-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 Why Long Context Matters

```
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT WINDOW = "Working Memory" of the LLM               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  4K tokens:   ~3 pages of text    (GPT-2, early LLaMA)     │
│  8K tokens:   ~6 pages            (LLaMA-2)                │
│  32K tokens:  ~25 pages           (GPT-4 base)             │
│  128K tokens: ~100 pages = a book (LLaMA-3, Claude 3)      │
│  1M tokens:   ~800 pages          (Gemini 1.5 Pro)         │
│  10M tokens:  ~8000 pages         (Gemini 2.0, research)   │
│                                                             │
│  APPLICATIONS that NEED long context:                       │
│  ┌──────────────────┬────────────────────────────────────┐ │
│  │ Task             │ Typical context needed             │ │
│  ├──────────────────┼────────────────────────────────────┤ │
│  │ Multi-turn chat  │ 4K - 32K                           │ │
│  │ Document QA      │ 8K - 128K                          │ │
│  │ Code repo        │ 64K - 1M                           │ │
│  │ Book analysis    │ 128K - 500K                        │ │
│  │ Video transcript │ 500K+                              │ │
│  └──────────────────┴────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The O(n²) Problem

```
ATTENTION COMPLEXITY:
  Time:   O(n² × d)     (n² score computations × d-dim vectors)
  Memory: O(n² × h)     (n² attention matrix per head)

FOR CONTEXT LENGTH n:
  n=4K:    16M entries per head  → 0.5 GB attention matrices
  n=32K:   1B entries per head   → 32 GB  ← exceeds GPU memory!
  n=128K:  16B entries per head  → 512 GB ← impossible naively!

QUADRATIC WALL:
  Doubling context → 4× memory and compute for attention.
  This is the fundamental bottleneck for long sequences.

SOLUTIONS:
  1. FlashAttention: O(n) memory, same O(n²) FLOPs but faster (tiling)
  2. Ring Attention: distribute n² across multiple GPUs
  3. Sparse attention: O(n × w) where w << n (approximate)
  4. Sliding window: only attend to last W tokens
```

### 1.3 Pipeline Summary

```
EXTENDING CONTEXT from L_train to L_target:

  Step 1: POSITION ENCODING FIX
    RoPE frequencies → modified to cover [0, L_target]
    Methods: PI, NTK, YaRN
      ↓
  Step 2: ATTENTION MEMORY FIX
    FlashAttention + Ring Attention for long sequences
    Or: sliding window for local attention
      ↓
  Step 3: CONTINUED PRETRAINING
    Train on long documents for 1000-10000 steps
    Use progressive length schedule: 4K → 16K → 64K → 128K
      ↓
  Step 4: KV CACHE MANAGEMENT
    PagedAttention + token eviction for inference
      ↓
  OUTPUT: model that handles L_target context effectively
```

---

## 2. Position Extrapolation Methods

### 2.1 Position Interpolation (PI)

```
IDEA: Compress all positions to fit within training range.

  ORIGINAL RoPE angle at position t, frequency i:
    angle = t × θᵢ    where θᵢ = 1/10000^{2i/d}
  
  AT TRAINING: max angle seen = L_train × θᵢ
  AT INFERENCE: position t > L_train → angle EXCEEDS training range!

  PI FIX: scale position by L_train/L_target
    angle_PI = (t × L_train / L_target) × θᵢ
  
  Now: max angle at t=L_target = L_train × θᵢ (same as training!) ✓

EXAMPLE (L_train=4096, L_target=32768):
  Scale factor = 4096/32768 = 1/8
  Position 32767 is treated as position 4095.9 → within trained range
  
LIMITATION:
  Nearby tokens (t and t+1) now have SMALLER angular difference:
    Original: Δangle = 1 × θᵢ
    PI:       Δangle = (1/8) × θᵢ   ← 8× less distinguishable!
  
  → Model needs 1000-2000 fine-tuning steps to adapt to denser angles.
```

### 2.2 NTK-Aware Scaling

```
INSIGHT: Don't treat all frequencies equally.
  HIGH-FREQ (local syntax): interpolate (compress)
  LOW-FREQ (global structure): extrapolate (keep unchanged)

MODIFY THE BASE instead of the position:
  θᵢ' = 1 / b'^{2i/d}
  
  where b' = b × (L_target/L_train)^{d/(d-2)}
  
  (b = 10000 is the original RoPE base)

EFFECT ON DIFFERENT FREQUENCIES:
  ┌──────────────────────────────────────────────────────────┐
  │  High frequency (i≈0):   θ₀' ≈ θ₀ × (L_train/L_target) │
  │    → Interpolated like PI (nearby tokens still distinct) │
  │                                                          │
  │  Low frequency (i≈d/2):  θ_max' ≈ θ_max                │
  │    → Unchanged (long-range patterns preserved)           │
  │                                                          │
  │  Middle frequencies: smoothly interpolated between        │
  └──────────────────────────────────────────────────────────┘

ADVANTAGE: works WITHOUT fine-tuning for up to 4× extension!
  (PI requires 1000+ steps of fine-tuning)
```

### 2.3 YaRN (Yet Another RoPE Extension)

```
YARN = NTK-by-parts + temperature scaling + attention scaling

COMPONENT 1: NTK-by-parts
  Divide frequencies into 3 categories:
    High freq (λᵢ < L_train): DON'T modify (already seen all angles)
    Low freq (λᵢ > L_target):  FULLY interpolate (like PI)
    Middle freq:               LINEAR ramp between no-change and full interp

COMPONENT 2: Temperature scaling
  After RoPE rotation, scale attention scores:
    score = (q·k) / (√d_k × t)    where t = √(L_target/L_train)
  
  This compensates for the increased entropy of attention
  at longer context lengths.

COMPONENT 3: Dynamic NTK (inference-time)
  Adjust base b' dynamically based on CURRENT sequence length:
    If current_len ≤ L_train: use original θ
    If current_len > L_train: apply scaling proportional to current_len

RESULTS:
  YaRN enables LLaMA-2 (trained at 4K) to work at 128K
  with only ~400 steps of fine-tuning.
  
  Perplexity at various context lengths (LLaMA-2 7B + YaRN):
    4K:   5.8  (original quality maintained)
    16K:  6.0  (minimal degradation)
    64K:  6.4  (moderate degradation)
    128K: 7.1  (usable but not ideal)
```

---

## 3. Efficient Attention for Long Sequences

### 3.1 Ring Attention

```
PROBLEM: n=128K with FlashAttention still needs huge KV cache on one GPU.

RING ATTENTION (Liu et al. 2023):
  Distribute the sequence across P GPUs in a ring topology.
  Each GPU holds n/P tokens.
  
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │GPU 0 │───►│GPU 1 │───►│GPU 2 │───►│GPU 3 │──┐
  │tok 0-│    │tok   │    │tok   │    │tok   │  │
  │  31K │    │32-63K│    │64-95K│    │96-127K│  │
  └──────┘    └──────┘    └──────┘    └──────┘  │
       ↑                                         │
       └─────────────────────────────────────────┘

  ALGORITHM:
  For each of P rounds:
    1. Each GPU computes attention between its Q and current K,V block
    2. Pass K,V block to next GPU in the ring (async communication)
    3. Accumulate attention output using online softmax
  
  After P rounds: each GPU has attended to ALL tokens!
  
  MEMORY PER GPU: O(n/P × d)  (only local Q + one KV block)
  COMMUNICATION: overlapped with computation (hidden latency)
  
  RESULT: can handle n = P × local_capacity
    4 GPUs × 32K local = 128K context ✓
    8 GPUs × 128K local = 1M context! ✓
```

### 3.2 Sparse Attention Patterns

```
INSTEAD OF FULL n×n ATTENTION, attend to only a SUBSET:

SLIDING WINDOW (Mistral, Longformer local):
  Each token attends only to W previous tokens.
  Complexity: O(n × W) instead of O(n²)
  
  For W=4096: position 10000 can only "see" positions 5904-10000.
  Information propagates via residual connections across layers:
    Layer 1: sees 4096 tokens back
    Layer 2: sees 8192 tokens back (via layer 1's output)
    Layer L: sees L×4096 tokens back (effective receptive field)

GLOBAL + LOCAL (Longformer):
  Most tokens: sliding window (local attention)
  Special [CLS] tokens: attend to EVERYTHING (global attention)
  
  Cost: O(n × W + n × G) where G = number of global tokens

DILATED (BigBird):
  Attend to every k-th token for long range + local window.
  Combines: local + dilated + random + global attention.
  
  Approximation quality: typically 95-99% of full attention
```

---

## 4. Memory Optimizations

### 4.1 Sliding Window Attention

```
MISTRAL (Jiang et al. 2023):
  Window size W = 4096
  
  ATTENTION MASK (n=8, W=4 for illustration):
       k=1  k=2  k=3  k=4  k=5  k=6  k=7  k=8
  q=1 [ ✓    ✗    ✗    ✗    ✗    ✗    ✗    ✗ ]
  q=2 [ ✓    ✓    ✗    ✗    ✗    ✗    ✗    ✗ ]
  q=3 [ ✓    ✓    ✓    ✗    ✗    ✗    ✗    ✗ ]
  q=4 [ ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗ ]
  q=5 [ ✗    ✓    ✓    ✓    ✓    ✗    ✗    ✗ ]  ← window slides
  q=6 [ ✗    ✗    ✓    ✓    ✓    ✓    ✗    ✗ ]
  q=7 [ ✗    ✗    ✗    ✓    ✓    ✓    ✓    ✗ ]
  q=8 [ ✗    ✗    ✗    ✗    ✓    ✓    ✓    ✓ ]

  KV CACHE SAVINGS:
    Full causal: store ALL past tokens = O(n)
    Sliding window: store only last W tokens = O(W) = constant!
    
    For n=128K, W=4096:
      Full KV cache: 128K × 128 KB = 16.4 GB
      Sliding KV:    4K × 128 KB = 0.5 GB  ← 32× savings!

  EFFECTIVE CONTEXT (via layering):
    1 layer:   W tokens back
    L layers:  L × W tokens of effective context
    LLaMA-3 (L=32, W=8192): effective = 32 × 8192 = 262K tokens
    → Full 128K context is within the effective receptive field ✓
```

### 4.2 Token Eviction and Compression

```
H₂O (Heavy-Hitter Oracle, Zhang et al. 2023):
  Not all tokens in the KV cache are equally important.
  Some are "heavy hitters" (high attention) that should be kept.
  Others can be evicted with minimal quality loss.

ALGORITHM:
  1. Track cumulative attention each KV entry receives
  2. When cache is full (exceeds budget B):
     - Always keep: first few tokens (anchor) + last W tokens (recent)
     - From middle: keep top-K by cumulative attention score
     - Evict the rest
  
  RESULT: O(B) memory where B << n, with minimal quality loss.
  
  Typical B: 1024-4096 (fixed budget regardless of context length!)
  Quality: < 1% PPL increase even for n >> B.
```

---

## 5. Training Long-Context Models

### 5.1 Progressive Length Training

```
WHY NOT TRAIN DIRECTLY ON LONG SEQUENCES?
  Long sequences = fewer sequences per batch = noisier gradients.
  Training on 128K from scratch is wasteful for early learning.

PROGRESSIVE SCHEDULE:
  Phase 1: L=4K   for 90% of training tokens  (efficient learning)
  Phase 2: L=16K  for 5% of tokens            (length adaptation)
  Phase 3: L=64K  for 3% of tokens
  Phase 4: L=128K for 2% of tokens            (final extension)

COMPUTE OVERHEAD:
  Long-context fine-tuning: only 0.1-1% of total pretraining FLOPs
  ~1000-10000 steps at the target length is sufficient.
```

### 5.2 Long-Context Data

```
DATA REQUIREMENTS:
  Need documents that are ACTUALLY long and coherent:
  - Books (fiction, textbooks)
  - Long code files (>10K lines)
  - Legal documents
  - Scientific papers
  - Multi-turn conversations
  - Concatenated related documents

SYNTHETIC LONG-CONTEXT DATA:
  "Needle in a haystack" tests:
    Insert a random fact in the middle of a long context.
    Ask the model to recall it.
    If the model can't: needs more long-context training.
```

---

## 6. Summary

### 6.1 Formulas Quick Reference

**Position Interpolation:**

```
angle_PI(t) = (t × L_train / L_target) × θᵢ
```

**NTK-Aware base:**

```
b' = b × (L_target / L_train)^{d/(d-2)}
```

**Sliding window KV cache size:**

```
M_KV = W × 2 × L × h_kv × d_k × sizeof(dtype)   (fixed, independent of n!)
```

**Ring Attention memory per GPU:**

```
M_per_gpu = (n/P) × (d + 2×L×h_kv×d_k) × sizeof(dtype)
```

| Method | Context Extension | Fine-tuning needed | Quality at 8× |
|--------|------------------|--------------------|----------------|
| PI | Arbitrary | 1000-2000 steps | Good |
| NTK-Aware | Up to 4× | None | Moderate |
| YaRN | Up to 32× | 400 steps | Good |
| Ring Attention | Unlimited (add GPUs) | None | Full quality |
| Sliding Window | Unlimited (evict old) | None | Approximate |

### 6.2 Common Mistakes

```
❌ WRONG: Just changing RoPE base extends context without quality loss
✓ RIGHT:  RoPE modifications fix the POSITION ENCODING extrapolation.
          But the model also needs to learn to USE long contexts effectively.
          Without continued pretraining on long data: the model ignores 
          distant context even if it can technically encode the positions.

❌ WRONG: 128K context = model uses all 128K tokens equally well
✓ RIGHT:  "Lost in the middle" effect: models attend strongly to the
          beginning and end, weakly to the middle. Context utilisation
          degrades with distance even within the supported length.

❌ WRONG: Sliding window attention loses all long-range information
✓ RIGHT:  Information propagates across layers through residual connections.
          Layer L effectively has receptive field of L×W tokens.
          For L=32, W=4096: effective context = 131K tokens.

❌ WRONG: Ring Attention is always necessary for long context
✓ RIGHT:  FlashAttention handles up to ~128K on a single A100 (80 GB)
          with GQA. Ring Attention is needed for 1M+ contexts or when
          KV cache exceeds single-GPU memory.
```

---

## 7. Exercises

1. **PI Calculation**: For L_train=8192 and L_target=65536, compute the PI scale factor. What angle does position 65535 map to? Compare to the original angle.

2. **Sliding Window KV Cache**: For W=4096, L=32, h_kv=8, d_k=128, BF16: compute the fixed KV cache size. Compare to full causal attention cache at n=128K.

3. **Ring Attention Memory**: For n=1M tokens, d=4096, P=8 GPUs, BF16: compute per-GPU memory for the sequence partition. Is this feasible on 80 GB GPUs?

4. **Effective Receptive Field**: A model has L=32 layers with sliding window W=4096. What is the maximum effective context length (L×W)? At position 100000, what is the earliest position the model can "see" information from?
