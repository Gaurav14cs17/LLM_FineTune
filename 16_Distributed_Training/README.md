# Understanding Distributed LLM Training

*Data, Tensor, Pipeline, and Expert Parallelism — the mathematics of splitting models across GPUs*

---

Training an 8B-parameter LLM requires 128+ GB of memory and 10²³ FLOPs. No single GPU can handle this. **Distributed training** splits the workload across dozens to thousands of GPUs, using four complementary parallelism strategies: Data Parallel (DP), Fully Sharded Data Parallel (FSDP/ZeRO), Tensor Parallel (TP), and Pipeline Parallel (PP).

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Distribute?](#11-why-distribute)
   - [1.2 The Four Parallelism Types](#12-the-four-parallelism-types)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Data Parallelism (DP)](#2-data-parallelism-dp)
   - [2.1 AllReduce Gradient Sync](#21-allreduce-gradient-sync)
   - [2.2 Scaling the Learning Rate](#22-scaling-the-learning-rate)
3. [ZeRO / FSDP](#3-zero--fsdp)
   - [3.1 ZeRO Stages 1, 2, 3](#31-zero-stages-1-2-3)
   - [3.2 Memory Savings Formula](#32-memory-savings-formula)
4. [Tensor Parallelism (TP)](#4-tensor-parallelism-tp)
   - [4.1 Column-Parallel Linear](#41-column-parallel-linear)
   - [4.2 Row-Parallel Linear](#42-row-parallel-linear)
   - [4.3 Attention Parallelism](#43-attention-parallelism)
5. [Pipeline Parallelism (PP)](#5-pipeline-parallelism-pp)
   - [5.1 Micro-Batching](#51-micro-batching)
   - [5.2 Bubble Overhead](#52-bubble-overhead)
6. [3D Parallelism](#6-3d-parallelism)
   - [6.1 Combining DP + TP + PP](#61-combining-dp--tp--pp)
7. [Summary](#7-summary)
   - [7.1 Formulas Quick Reference](#71-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 Why Distribute?

```
┌─────────────────────────────────────────────────────────────┐
│  LLAMA-3 8B TRAINING REQUIREMENTS:                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Memory per GPU (full fine-tuning):                         │
│    BF16 weights:   16 GB                                    │
│    BF16 gradients: 16 GB                                    │
│    FP32 master:    32 GB                                    │
│    Adam states:    64 GB                                    │
│    Activations:    varies (10-50 GB per micro-batch)        │
│    TOTAL:          138-178 GB  ← NO single GPU can fit!    │
│                                                             │
│  Compute requirement:                                       │
│    C = 6 × 8B × 15T = 7.2×10²³ FLOPs                       │
│    Single A100 (312 TFLOPS BF16): 7.2e23/312e12/3600       │
│    = 641,000 GPU-hours = 73 GPU-years                      │
│    With 1024 GPUs: ~26 days (actual Meta training time)     │
│                                                             │
│  DISTRIBUTION IS MANDATORY, not optional!                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Four Parallelism Types

```
┌─────────────────────────────────────────────────────────────┐
│  DATA PARALLEL (DP): same model on every GPU, different data │
│                                                             │
│  GPU 0: Model copy → batch₀ → grad₀ ─┐                     │
│  GPU 1: Model copy → batch₁ → grad₁ ─┼→ AllReduce → avg grad│
│  GPU 2: Model copy → batch₂ → grad₂ ─┘                     │
│                                                             │
│  Scales: batch size (throughput). Doesn't help memory.      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TENSOR PARALLEL (TP): split each layer ACROSS GPUs         │
│                                                             │
│  Layer W ∈ ℝ^{d×d}:                                         │
│  GPU 0: W[:, :d/2]  (left half of columns)                  │
│  GPU 1: W[:, d/2:]  (right half of columns)                 │
│                                                             │
│  Each GPU computes PART of the layer, then AllReduce.       │
│  Scales: per-layer memory. Needs fast interconnect.         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PIPELINE PARALLEL (PP): split layers across GPUs            │
│                                                             │
│  GPU 0: Layers 0-7                                          │
│  GPU 1: Layers 8-15                                         │
│  GPU 2: Layers 16-23                                        │
│  GPU 3: Layers 24-31                                        │
│                                                             │
│  Data flows sequentially: GPU0 → GPU1 → GPU2 → GPU3        │
│  Problem: "pipeline bubbles" when GPUs wait.               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  FSDP/ZeRO: shard model + optimizer across DP ranks         │
│                                                             │
│  Each GPU stores 1/N of weights + optimizer states.         │
│  AllGather weights on demand, compute, discard after.       │
│  Scales: memory per GPU (reduces by factor N).             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Pipeline Summary

```
TYPICAL LARGE-SCALE CONFIG (LLaMA-3 8B pretraining):

  Total GPUs: 1024  (128 nodes × 8 GPUs/node)
  
  DP (data parallel): 128 groups → global batch = 128 × micro_batch
  TP (tensor parallel): 2 GPUs per group (within node, NVLink)
  PP (pipeline parallel): 4 stages (across nodes)
  
  Each GPU handles:
    1/(TP×PP) of model params = 1/8 of 8B = 1B params per GPU
    1/DP of data = 1/128 of global batch
```

---

## 2. Data Parallelism (DP)

### 2.1 AllReduce Gradient Sync

```
After each micro-batch, gradients must be AVERAGED across all DP ranks.

RING-ALLREDUCE (bandwidth-optimal for N GPUs):

  Step 1: Reduce-Scatter
    Each GPU sends 1/N of its gradient to the "responsible" GPU.
    After N-1 rounds: each GPU has the SUM of 1/N of all gradients.
    
  Step 2: AllGather  
    Each GPU broadcasts its completed chunk to all others.
    After N-1 rounds: every GPU has the full averaged gradient.

COMMUNICATION VOLUME:
  Total data per GPU: 2 × (N-1)/N × |params| × sizeof(grad)
  ≈ 2 × |params| × sizeof(grad)   (for large N)
  
  For LLaMA-3 8B BF16:
    2 × 8B × 2 bytes = 32 GB per AllReduce
    With NVLink 900 GB/s (intra-node): ~36 ms
    With InfiniBand 400 Gb/s (inter-node): ~640 ms
```

### 2.2 Scaling the Learning Rate

```
LINEAR SCALING RULE (Goyal et al. 2017):

  If global batch B_total = B_per_gpu × N_gpus:
    η_effective = η_base × N_gpus

  WHY? The gradient variance DECREASES by 1/N with N replicas.
    Var[avg_grad] = Var[grad] / N
    We can take larger steps because gradient estimate is more reliable.
    
  WARMUP IS CRITICAL:
    Start with η_base (no scaling), warmup to η_effective over ~2000 steps.
    Without warmup: model diverges at the start (large LR + poor statistics).
```

---

## 3. ZeRO / FSDP

### 3.1 ZeRO Stages 1, 2, 3

```
MEMORY PER GPU (8B model, N=8 GPUs, BF16 training):

┌────────────────────────────────────────────────────────────────┐
│  Stage    Shards what?              Memory/GPU   Savings      │
├────────────────────────────────────────────────────────────────┤
│  None     Nothing (vanilla DP)       128 GB      baseline     │
│  Stage 1  Optimizer states (m, v)     48 GB      2.7×         │
│  Stage 2  + Gradients                 32 GB      4×           │
│  Stage 3  + Parameters (= FSDP)       16+α GB   8×           │
│                                       (α = activation memory) │
└────────────────────────────────────────────────────────────────┘

ZeRO-3 / FSDP MECHANISM:
  ┌─────────────────────────────────────────────────────────────┐
  │  FORWARD PASS for layer l:                                  │
  │  1. AllGather: collect full W_l from all GPUs (each has 1/N)│
  │  2. Compute: y = f(x, W_l)                                 │
  │  3. Discard: free W_l memory (only keep local 1/N shard)   │
  │                                                             │
  │  BACKWARD PASS for layer l:                                 │
  │  1. AllGather: collect full W_l again                       │
  │  2. Compute: ∂L/∂x, ∂L/∂W_l                               │
  │  3. ReduceScatter: average gradient, keep only local shard │
  │  4. Discard: free W_l memory                               │
  └─────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Savings Formula

```
MEMORY PER GPU (dense model, ZeRO-3, BF16 training):

  M_per_gpu = (N_params / N_gpus) × bytes_per_param × 
              (1 + 1 + 2 + 4 + 4)   
              ↑weight ↑grad ↑master ↑m₁  ↑m₂
              
  Simplified:
  M_per_gpu ≈ 12 × N_params / N_gpus  (bytes, mixed precision)
  
  For 8B model, 8 GPUs:
    M_per_gpu = 12 × 8B / 8 = 12 GB for params+optimizer
    + activations (varies with batch size and recomputation)
  
  COMPARE to vanilla DP:
    M_per_gpu = 12 × 8B = 96 GB + 16 GB weights = 112 GB!
  
  ZeRO-3 SAVINGS: 112 / 12 ≈ 9×  (can train 9× larger models!)
```

---

## 4. Tensor Parallelism (TP)

### 4.1 Column-Parallel Linear

```
SPLITTING: Y = X × W, split W by COLUMNS across T GPUs.

  W ∈ ℝ^{d×k} split into W = [W₁ | W₂ | ... | Wₜ]
  Each GPU i holds W_i ∈ ℝ^{d×(k/T)}

  GPU i computes: Y_i = X × W_i ∈ ℝ^{n×(k/T)}

  Result: Y = [Y₁ | Y₂ | ... | Yₜ]  (concatenated along columns)

COMMUNICATION:
  Forward: no communication needed (each GPU has full X)
  Backward: AllReduce on ∂L/∂X (each GPU has partial gradient)

VISUALISATION (T=2 GPUs, d=4, k=4):
  W = [w₁₁ w₁₂ | w₁₃ w₁₄]    GPU 0 has [w₁₁ w₁₂]
      [w₂₁ w₂₂ | w₂₃ w₂₄]    GPU 1 has [w₁₃ w₁₄]
      [w₃₁ w₃₂ | w₃₃ w₃₄]              [w₂₃ w₂₄]
      [w₄₁ w₄₂ | w₄₃ w₄₄]              [w₃₃ w₃₄]
                                          [w₄₃ w₄₄]
```

### 4.2 Row-Parallel Linear

```
SPLITTING: Y = X × W, split W by ROWS.

  W ∈ ℝ^{d×k} split into W = [W₁; W₂; ...; Wₜ]  (row blocks)
  Each GPU i holds W_i ∈ ℝ^{(d/T)×k}
  Input X must also be split: X_i ∈ ℝ^{n×(d/T)}

  GPU i computes: Y_i = X_i × W_i ∈ ℝ^{n×k}

  Result: Y = Y₁ + Y₂ + ... + Yₜ  (AllReduce SUM)

COMMUNICATION:
  Forward: AllReduce on Y (sum partial results)
  Backward: no communication (each GPU gets full ∂L/∂Y)
```

### 4.3 Attention Parallelism

```
MULTI-HEAD ATTENTION — natural TP split:

  Heads 0-15: GPU 0
  Heads 16-31: GPU 1

  Each GPU computes attention for its own heads independently!
  Only need AllReduce after the output projection W_O.

COMMUNICATION PATTERN per transformer layer (TP=2):
  ┌────────────────────────────────────────────────────────────┐
  │  Input x (both GPUs have full x via AllReduce)            │
  │      ↓                                                     │
  │  Column-parallel: Q,K,V projections (no comm)             │
  │      ↓                                                     │
  │  Attention (per-head, no comm)                             │
  │      ↓                                                     │
  │  Row-parallel: output projection → AllReduce              │
  │      ↓                                                     │
  │  Column-parallel: FFN gate_proj, up_proj (no comm)         │
  │      ↓                                                     │
  │  Row-parallel: FFN down_proj → AllReduce                   │
  │      ↓                                                     │
  │  Output (both GPUs have full output)                       │
  │                                                            │
  │  TOTAL: 2 AllReduces per transformer layer                 │
  │  Volume: 2 × 2 × n × d × sizeof(BF16) per layer           │
  │  For n=4096, d=4096: 2 × 2 × 4096 × 4096 × 2 = 128 MB    │
  └────────────────────────────────────────────────────────────┘
```

---

## 5. Pipeline Parallelism (PP)

### 5.1 Micro-Batching

```
PROBLEM: with PP, GPUs sit idle waiting for data from previous stage.

  GPU 0: [compute]...[idle]..........[compute]
  GPU 1: [idle]...[compute]...[idle].........
  GPU 2: [idle].........[compute]...[idle]...
  GPU 3: [idle]...............[compute].......

SOLUTION: split batch into M micro-batches:

  GPU 0: [μ1][μ2][μ3][μ4]...............
  GPU 1: ....[μ1][μ2][μ3][μ4]...........
  GPU 2: ........[μ1][μ2][μ3][μ4].......
  GPU 3: ............[μ1][μ2][μ3][μ4]...

  The pipeline stays FULL most of the time!
  Idle time = "bubble" at start and end only.
```

### 5.2 Bubble Overhead

```
BUBBLE FRACTION:

  P = number of pipeline stages
  M = number of micro-batches
  
  Bubble fraction = (P - 1) / (P - 1 + M)
  
  For P=4, M=32: bubble = 3/35 = 8.6% (acceptable!)
  For P=4, M=4:  bubble = 3/7 = 42.9%  (too much waste!)

RULE OF THUMB: M ≥ 4×P for ≤20% bubble overhead.

ADVANCED: 1F1B SCHEDULE (one forward, one backward)
  Instead of all-forwards-then-all-backwards:
  Interleave F and B passes to reduce peak activation memory.
  
  Memory per stage: only need to store activations for P micro-batches
  (not all M), reducing activation memory by M/P.
```

---

## 6. 3D Parallelism

### 6.1 Combining DP + TP + PP

```
EXAMPLE: 128 GPUs training 70B model

  TP = 4  (split each layer across 4 GPUs, within one node)
  PP = 8  (split 80 layers into 8 stages of 10 layers each)
  DP = 4  (4 data-parallel replicas)
  
  Total: 4 × 8 × 4 = 128 GPUs ✓

PLACEMENT:
  TP group:  same node, connected via NVLink (900 GB/s)
  PP group:  across nodes, connected via InfiniBand (400 Gb/s)
  DP group:  any GPUs, asynchronous AllReduce
  
  ┌─────── Node 0 ────────┐   ┌─────── Node 1 ────────┐
  │ GPU0 GPU1 GPU2 GPU3    │   │ GPU0 GPU1 GPU2 GPU3    │
  │ ←── TP group ────→    │   │ ←── TP group ────→    │
  │ PP stage 0             │   │ PP stage 1             │
  └──────────┬─────────────┘   └──────────┬─────────────┘
             └── PP connection (InfiniBand) ──┘

THROUGHPUT FORMULA:
  Tokens/sec = (DP × micro_batch_size × seq_len) / time_per_step
  
  For LLaMA-3 8B (128 GPUs, as configured):
    Global batch: 4 × 4 × 4096 = 65,536 tokens per micro-batch
    ~3000 tokens/sec/GPU → 384,000 tokens/sec total
    Time to train on 15T tokens: 15T / 384K / 3600 ≈ 10,850 hours ≈ 452 days
    (Actual: with 1024 GPUs and better config, ~26 days)
```

---

## 7. Summary

### 7.1 Formulas Quick Reference

**ZeRO-3 memory per GPU:**

```
M = (12 × N / N_gpus) + M_activations
```

**AllReduce volume (Ring):**

```
V = 2 × (N-1)/N × |params| × bytes_per_value ≈ 2 × |params| × bytes
```

**Pipeline bubble:**

```
Bubble% = (P-1) / (P-1+M) × 100%
```

**Tensor parallel communication per layer:**

```
V_TP = 2 × 2 × n × d × sizeof(dtype)   (2 AllReduces, factor 2 for ring)
```

| Strategy | Splits | Scales | Requires |
|----------|--------|--------|----------|
| DP | Batch | Throughput | AllReduce |
| ZeRO/FSDP | Params+opt | Memory | AllGather + ReduceScatter |
| TP | Layer weights | Per-layer memory | Fast interconnect (NVLink) |
| PP | Layers | Total model | Micro-batching |

### 7.2 Common Mistakes

```
❌ WRONG: More GPUs always means linearly faster training
✓ RIGHT:  Communication overhead grows with GPU count. At some point,
          GPUs spend more time communicating than computing.
          Typical scaling efficiency: 70-90% at 1000 GPUs.

❌ WRONG: TP=8 is better than TP=2 (more split = more memory savings)
✓ RIGHT:  TP communication happens at EVERY layer (2 AllReduces/layer).
          TP=8 means 8-way AllReduce at every layer — very expensive
          unless GPUs are connected by NVLink. Never use TP across nodes!

❌ WRONG: ZeRO-3 eliminates all memory redundancy
✓ RIGHT:  ZeRO-3 eliminates redundancy in PARAMETERS and OPTIMIZER STATES.
          Activations are still replicated (each GPU needs them for backward).
          Activation checkpointing (recomputation) handles activation memory.

❌ WRONG: Pipeline parallelism reduces per-GPU memory by factor P
✓ RIGHT:  PP reduces MODEL memory by factor P (each GPU has L/P layers).
          But each stage must store activations for M micro-batches.
          Total per-GPU memory = model/P + activations×min(M,P).
```

---

## 8. Exercises

1. **ZeRO Memory**: For a 70B model with 128 GPUs, ZeRO-3: compute per-GPU memory for parameters + optimizer states. If activations are 20 GB per GPU, what's total per-GPU memory?

2. **Pipeline Bubble**: You have P=8 stages and want bubble < 10%. What is the minimum number of micro-batches M needed?

3. **TP Communication**: For TP=4, d=4096, seq_len=4096, BF16: compute communication volume per transformer layer. If NVLink bandwidth is 900 GB/s, what is the communication time? Compare to computation time (assume 312 TFLOPS).

4. **3D Parallelism Config**: You have 256 GPUs (32 nodes × 8 GPUs). Model: 175B params, 96 layers. Design a 3D parallelism config (choose TP, PP, DP) that minimises communication while keeping per-GPU memory under 80 GB.
