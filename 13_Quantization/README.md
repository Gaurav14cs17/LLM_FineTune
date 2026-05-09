# Understanding LLM Quantization

*From FP32 to INT4 — the mathematics of compressing models with minimal quality loss*

---

**Quantization** reduces model size and inference cost by representing weights (and optionally activations) in lower precision. A BF16 model uses 2 bytes per parameter; INT4 uses 0.5 bytes — a 4× reduction. This makes it possible to run LLaMA-3 70B on a single consumer GPU.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 Why Quantize?](#11-why-quantize)
   - [1.2 Number Formats at a Glance](#12-number-formats-at-a-glance)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Uniform Quantization](#2-uniform-quantization)
   - [2.1 Symmetric Quantization](#21-symmetric-quantization)
   - [2.2 Asymmetric Quantization](#22-asymmetric-quantization)
   - [2.3 Quantization Error Analysis](#23-quantization-error-analysis)
3. [Group Quantization](#3-group-quantization)
   - [3.1 Per-Channel vs Per-Group](#31-per-channel-vs-per-group)
4. [Advanced Methods](#4-advanced-methods)
   - [4.1 GPTQ (Post-Training, Weight-Only)](#41-gptq-post-training-weight-only)
   - [4.2 AWQ (Activation-Aware Weights)](#42-awq-activation-aware-weights)
   - [4.3 NF4 (Normal Float 4-bit — QLoRA)](#43-nf4-normal-float-4-bit--qlora)
5. [Activation Quantization](#5-activation-quantization)
   - [5.1 SmoothQuant](#51-smoothquant)
   - [5.2 INT8 KV Cache](#52-int8-kv-cache)
6. [Memory Savings Calculations](#6-memory-savings-calculations)
   - [6.1 LLaMA-3 8B at Various Precisions](#61-llama-3-8b-at-various-precisions)
7. [Summary](#7-summary)
   - [7.1 Formulas Quick Reference](#71-formulas-quick-reference)
   - [7.2 Method Comparison Table](#72-method-comparison-table)
   - [7.3 Common Mistakes](#73-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 Why Quantize?

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL SIZE vs PRECISION                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLaMA-3 70B model weight memory:                           │
│    FP32:   70B × 4 bytes = 280 GB  (4× A100 80GB!)         │
│    BF16:   70B × 2 bytes = 140 GB  (2× A100)               │
│    INT8:   70B × 1 byte  = 70 GB   (1× A100)               │
│    INT4:   70B × 0.5 B   = 35 GB   (1× RTX 4090!) ✓       │
│                                                             │
│  ADDITIONALLY: lower precision → faster compute             │
│    INT8 matmul: 2× faster than BF16 on modern GPUs         │
│    INT4 matmul: 4× faster (with hardware support)          │
│                                                             │
│  QUALITY IMPACT (perplexity increase):                      │
│    BF16 → INT8:  < 0.1% PPL increase   (negligible)        │
│    BF16 → INT4:  ~ 0.5-1% PPL increase (acceptable)       │
│    BF16 → INT3:  ~ 3-5% PPL increase   (noticeable)       │
│    BF16 → INT2:  > 10% PPL increase    (significant)       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Number Formats at a Glance

```
FP32 (32 bits): [1 sign | 8 exponent | 23 mantissa]
  Range: ±3.4×10³⁸,  Precision: ~7 decimal digits
  
BF16 (16 bits): [1 sign | 8 exponent | 7 mantissa]
  Range: same as FP32!  Precision: ~3 decimal digits
  (used for TRAINING — range matters more than precision)
  
FP16 (16 bits): [1 sign | 5 exponent | 10 mantissa]
  Range: ±65,504    Precision: ~4 decimal digits
  (can overflow! that's why BF16 is preferred for LLMs)

INT8 (8 bits):  [1 sign | 7 magnitude]  or unsigned [8 bits]
  Range: -128 to 127 (signed) or 0 to 255 (unsigned)
  Only 256 possible values!
  
INT4 (4 bits):  [4 bits]
  Range: 0 to 15 (unsigned) or -8 to 7 (signed)
  Only 16 possible values!

┌──────┬──────┬─────────────────┬─────────────────────────────┐
│ Type │ Bits │ Bytes/param     │ 8B model size               │
├──────┼──────┼─────────────────┼─────────────────────────────┤
│ FP32 │  32  │ 4.0             │ 32 GB                       │
│ BF16 │  16  │ 2.0             │ 16 GB                       │
│ INT8 │   8  │ 1.0             │ 8 GB                        │
│ INT4 │   4  │ 0.5             │ 4 GB                        │
│ NF4  │   4  │ 0.5 + overhead  │ ~4.5 GB (with group scales) │
└──────┴──────┴─────────────────┴─────────────────────────────┘
```

### 1.3 Pipeline Summary

```
POST-TRAINING QUANTIZATION (PTQ — most common for deployment):

  Trained BF16 model
      ↓
  Calibration: run ~128 samples through model, collect statistics
      ↓
  Determine scale/zero-point for each weight group
      ↓
  Quantize: w_int = round(w_fp / scale + zero_point)
      ↓
  Store: INT4 weights + FP16 scales (per group)
      ↓
  Inference: dequantize on-the-fly during matmul
      w_fp ≈ (w_int - zero_point) × scale
```

---

## 2. Uniform Quantization

### 2.1 Symmetric Quantization

```
SYMMETRIC: zero maps to zero, range is symmetric around 0.

  QUANTIZE:
    scale = max(|w|) / (2^{b-1} - 1)
    w_int = round(w / scale)         clamp to [-2^{b-1}, 2^{b-1}-1]
  
  DEQUANTIZE:
    ŵ = w_int × scale

  EXAMPLE (INT8, b=8):
    Weights: [-0.8, 0.3, -1.2, 0.7, 1.5]
    max|w| = 1.5
    scale = 1.5 / 127 = 0.0118
    
    w_int = round([-0.8, 0.3, -1.2, 0.7, 1.5] / 0.0118)
          = round([-67.8, 25.4, -101.7, 59.3, 127.0])
          = [-68, 25, -102, 59, 127]
    
    Dequantized: [-68, 25, -102, 59, 127] × 0.0118
               = [-0.802, 0.295, -1.204, 0.696, 1.499]
    
    Error:       [0.002, -0.005, 0.004, -0.004, -0.001]
    Max error:   0.005 (half a quantization step)
```

### 2.2 Asymmetric Quantization

```
ASYMMETRIC: maps [w_min, w_max] to [0, 2^b - 1] (better for non-symmetric)

  QUANTIZE:
    scale = (w_max - w_min) / (2^b - 1)
    zero_point = round(-w_min / scale)
    w_int = round(w / scale) + zero_point   clamp to [0, 2^b - 1]

  DEQUANTIZE:
    ŵ = (w_int - zero_point) × scale

  EXAMPLE (INT4 unsigned, b=4):
    Weights: [0.1, 0.3, 0.8, 1.2, 0.5]
    w_min = 0.1, w_max = 1.2
    scale = (1.2 - 0.1) / (16 - 1) = 1.1/15 = 0.0733
    zero_point = round(-0.1 / 0.0733) = round(-1.36) = -1
    
    w_int = round([0.1, 0.3, 0.8, 1.2, 0.5] / 0.0733) + (-1)
          = round([1.36, 4.09, 10.9, 16.4, 6.82]) - 1
          = [1, 4, 11, 15, 7] - 1 = [0, 3, 10, 14, 6]    (clamped to [0,15])
    
    Dequantized: ([0, 3, 10, 14, 6] - (-1)) × 0.0733
               = [1, 4, 11, 15, 7] × 0.0733
               = [0.073, 0.293, 0.807, 1.100, 0.513]
```

### 2.3 Quantization Error Analysis

```
┌─────────────────────────────────────────────────────────────┐
│  QUANTIZATION ERROR BOUND (uniform quantization):           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For b-bit symmetric quantization:                          │
│    step_size = 2 × max|w| / (2^b - 1)                      │
│    max_error = step_size / 2                                │
│                                                             │
│  For weights ~ N(0, σ²):                                    │
│    max|w| ≈ 3σ (99.7% coverage)                             │
│    step_size ≈ 6σ / (2^b - 1)                              │
│    RMS error ≈ step_size / √12  (uniform quantization noise)│
│                                                             │
│  RELATIVE error (Signal-to-Quantization-Noise Ratio):      │
│    SQNR ≈ 6.02 × b + 4.77 dB  (for Gaussian weights)      │
│                                                             │
│    INT8:  SQNR ≈ 53 dB → 0.2% relative error              │
│    INT4:  SQNR ≈ 29 dB → 3.5% relative error              │
│    INT3:  SQNR ≈ 23 dB → 7% relative error                │
│    INT2:  SQNR ≈ 17 dB → 14% relative error               │
│                                                             │
│  → INT4 is the sweet spot: acceptable error, 4× compression│
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Group Quantization

### 3.1 Per-Channel vs Per-Group

```
PROBLEM: one scale per entire matrix is too coarse.

  If matrix has outlier channels (values 10× larger than typical):
    scale is dominated by outlier → all OTHER values quantized poorly.

SOLUTION: use separate scale per GROUP of values.

PER-TENSOR (1 scale for entire matrix):
  W ∈ ℝ^{4096×4096} → 1 scale + 1 zero_point
  Quality: poor (outliers dominate)

PER-CHANNEL (1 scale per output dimension):
  W ∈ ℝ^{4096×4096} → 4096 scales
  Quality: good (each channel calibrated independently)

PER-GROUP (1 scale per G consecutive values, G=128 typical):
  W ∈ ℝ^{4096×4096} → 4096×4096/128 = 131,072 scales
  Quality: best (fine-grained calibration)
  Overhead: 131K × 2 bytes = 256 KB per matrix (negligible)

┌─────────────────────────────────────────────────────────────┐
│  PER-GROUP QUANTIZATION (G=128):                            │
│                                                             │
│  Weight matrix row: [w₁, w₂, ..., w₁₂₈ | w₁₂₉, ..., w₂₅₆ | ...]│
│                      └───── group 1 ─────┘└──── group 2 ────┘     │
│                      scale₁, zp₁          scale₂, zp₂            │
│                                                             │
│  Each group of 128 values gets its own (scale, zero_point)  │
│  → Outliers only affect their own group, not the full row   │
│  → Quality approaches per-element scaling                   │
│                                                             │
│  EFFECTIVE BITS (including overhead of FP16 scales):         │
│    4 bits/value + 16 bits / G values = 4 + 16/128 = 4.125 bits │
│    Overhead: only 3% for G=128!                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Advanced Methods

### 4.1 GPTQ (Post-Training, Weight-Only)

```
GPTQ (Frantar et al. 2022):
  Quantize weights ONE COLUMN at a time.
  After quantizing column j: adjust remaining columns to COMPENSATE.
  
ALGORITHM:
  For j = 1 to d:
    1. Quantize column j:  W_q[:,j] = Quantize(W[:,j])
    2. Error: δ = W[:,j] − W_q[:,j]
    3. Compensate remaining columns:
       W[:, j+1:] += δ × H⁻¹[j, j+1:] / H⁻¹[j,j]
       
       where H = XᵀX (Hessian from calibration data)

KEY INSIGHT: the Hessian H tells us which weight perturbations 
hurt loss the most. GPTQ distributes quantization error to 
directions that hurt least.

QUALITY: INT4 GPTQ ≈ 0.5-1% PPL increase on LLaMA-2 7B
SPEED: quantization takes ~30 minutes for 7B model
```

### 4.2 AWQ (Activation-Aware Weights)

```
AWQ (Lin et al. 2023):
  Not all weights are equally important!
  Weights multiplied by LARGE activations matter more.

OBSERVATION:
  y = Wx
  If x_j is always large (a "salient channel"):
    → W[:,j] × x_j contributes more to output
    → Quantization error in W[:,j] is amplified by x_j!

SOLUTION: scale salient weight channels UP before quantizing.
  W̃[:,j] = W[:,j] × s_j       (scale up important channels)
  x̃_j = x_j / s_j             (scale down corresponding activations)
  
  Result: W̃x̃ = Wx (mathematically equivalent!)
  But: W̃[:,j] is larger → gets more quantization precision

CHOOSING s_j:
  s_j ∝ (average |x_j|)^α   where α ∈ [0, 1] (hyperparameter)
  
  α=0: no scaling (vanilla quantization)
  α=1: full activation-proportional scaling
  α=0.5: empirically best for most models

RESULT: INT4 AWQ < 0.3% PPL increase — BEST weight-only INT4 method!
```

### 4.3 NF4 (Normal Float 4-bit — QLoRA)

```
┌─────────────────────────────────────────────────────────────┐
│  NF4 KEY INSIGHT: LLM weights are approximately Gaussian!   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  UNIFORM INT4: equally spaced quantization levels           │
│    Levels: -8, -6, -4, -2, 0, 2, 4, 6  (uniform spacing)  │
│    Problem: Gaussian weights cluster near 0 — we waste      │
│    precision in the tails where few values exist!           │
│                                                             │
│  NF4: quantization levels match Gaussian distribution       │
│    More levels near 0 (where density is highest)           │
│    Fewer levels in tails (where density is lowest)          │
│                                                             │
│  CONSTRUCTION:                                              │
│  1. Normalise weights to N(0,1)                             │
│  2. Divide cumulative Gaussian into 16 equal-probability bins│
│  3. Quantization level = midpoint of each bin               │
│                                                             │
│  NF4 quantization levels (16 values for 4 bits):           │
│  [-1.0, -0.6962, -0.5251, -0.3949, -0.2844,              │
│   -0.1848, -0.0911, 0.0, 0.0796, 0.1609,                │
│    0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]          │
│                                                             │
│  Note: levels are DENSER near 0 → less error where          │
│  most weights live!                                         │
└─────────────────────────────────────────────────────────────┘

INFORMATION-THEORETIC OPTIMALITY:
  For data ~ N(0,σ²), NF4 minimises expected quantization error.
  Proof: uniform-probability bins maximise entropy of quantized values
  → each 4-bit code is equally likely → maximum information content.
```

---

## 5. Activation Quantization

### 5.1 SmoothQuant

```
PROBLEM: activations have OUTLIER CHANNELS (values 100× larger than typical).

  Typical activation: [-0.1, 0.3, -0.2, 0.1, 50.0, -0.3, 0.2, 60.0]
                                                    ↑              ↑ outliers!

  If we quantize this to INT8:
    scale = max|x| / 127 = 60/127 = 0.472
    Most values map to: round(0.3/0.472) = round(0.635) = 1
    → Catastrophic precision loss for the 99% of values near 0!

SMOOTHQUANT SOLUTION (Xiao et al. 2023):
  Migrate quantization difficulty from activations to weights.
  
  Y = X × W
    = (X × diag(s)⁻¹) × (diag(s) × W)
    = X̃ × W̃
  
  Choose s to SMOOTH the activation outliers:
    s_j = max|X[:,j]|^α / max|W[j,:]|^{1-α}    (α=0.5)
  
  X̃[:,j] = X[:,j] / s_j   → outliers reduced
  W̃[j,:] = W[j,:] × s_j   → absorbs the outlier scale

  After smoothing: both X̃ and W̃ are quantization-friendly!
  → Can do W8A8 (both weights and activations in INT8)
  → 2× speedup using INT8 tensor cores on A100/H100
```

### 5.2 INT8 KV Cache

```
KV CACHE QUANTIZATION:
  Standard: KV cache in BF16 → 128 KB per token (LLaMA-3 8B)
  Quantized: KV cache in INT8 → 64 KB per token (2× savings!)

  For 128K context:
    BF16: 128 KB × 128K = 16.4 GB
    INT8: 64 KB × 128K = 8.2 GB  ← fits on one GPU!

METHOD:
  Per-token quantization (each token's K/V quantized independently):
    K_int8[t] = round(K_bf16[t] / scale_k[t])
    V_int8[t] = round(V_bf16[t] / scale_v[t])
  
  One FP16 scale per token per head per layer.
  Overhead: negligible (32 × 8 × 2 bytes × 128K = 65 MB for all scales)

QUALITY IMPACT:
  KV cache INT8: < 0.1% PPL increase (nearly lossless)
  KV cache INT4: 0.5-2% PPL increase (depends on task)
```

---

## 6. Memory Savings Calculations

### 6.1 LLaMA-3 8B at Various Precisions

```
┌────────────────────────────────────────────────────────────────┐
│  MEMORY BREAKDOWN: LLaMA-3 8B inference (batch=1, ctx=4096)   │
├──────────────────┬────────┬────────┬────────┬────────┬────────┤
│  Component        │ FP32   │ BF16   │ INT8   │ INT4   │ NF4+D │
├──────────────────┼────────┼────────┼────────┼────────┼────────┤
│  Model weights   │ 32.0GB │ 16.0GB │ 8.0GB  │ 4.0GB  │ 4.5GB │
│  KV cache (BF16) │  0.5GB │  0.5GB │  0.5GB │  0.5GB │  0.5GB│
│  Activations     │  ~1GB  │  ~1GB  │  ~1GB  │  ~1GB  │  ~1GB │
│  ────────────────┼────────┼────────┼────────┼────────┼────────┤
│  TOTAL           │ 33.5GB │ 17.5GB │  9.5GB │  5.5GB │  6.0GB│
│  Fits on         │ A100   │ RTX4090│ RTX3090│ RTX3060│ RTX3060│
└──────────────────┴────────┴────────┴────────┴────────┴────────┘

NF4+D = NF4 with double quantization (QLoRA):
  - Weights in NF4 (4 bits)
  - Group scales in INT8 (not FP16) → further 0.5 GB savings
  - Double quantization = quantize the quantization scales!
```

---

## 7. Summary

### 7.1 Formulas Quick Reference

**Symmetric quantization:**

```
scale = max|w| / (2^{b-1} - 1)
w_int = round(w / scale)
ŵ = w_int × scale
```

**Asymmetric quantization:**

```
scale = (w_max - w_min) / (2^b - 1)
zero_point = round(-w_min / scale)
w_int = round(w / scale) + zero_point
ŵ = (w_int - zero_point) × scale
```

**Group quantization effective bits:**

```
effective_bits = b + sizeof(scale) × 8 / G
  e.g., INT4 with FP16 scales, G=128: 4 + 16/128 = 4.125 bits
```

### 7.2 Method Comparison Table

| Method | Bits | Calibration | Quality (PPL↑) | Speed |
|--------|------|-------------|----------------|-------|
| Round-to-Nearest | 4 | None | 1-3% | Fast |
| GPTQ | 4 | ~128 samples | 0.5-1% | 30 min |
| AWQ | 4 | ~128 samples | 0.2-0.5% | 10 min |
| NF4 (QLoRA) | 4 | None (analytical) | 0.3-0.8% | Instant |
| SmoothQuant | W8A8 | Calibration | < 0.2% | 2× speedup |

### 7.3 Common Mistakes

```
❌ WRONG: INT4 quantization makes inference 4× faster
✓ RIGHT:  Weight-only INT4 helps MEMORY (4× smaller), not always speed.
          Speed depends on whether compute is memory-bound (batch=1) or
          compute-bound (large batch). At batch=1 it's ~2× faster.
          For W8A8 (weights AND activations INT8): true 2× speedup.

❌ WRONG: Quantization is lossless if you keep enough bits
✓ RIGHT:  Quantization is ALWAYS lossy (except at original precision).
          The question is whether the loss is small enough to ignore.
          INT8 for weights: practically lossless.
          INT4 for weights: small loss, acceptable for most uses.

❌ WRONG: GPTQ is always better than round-to-nearest (RTN)
✓ RIGHT:  GPTQ is better for INT4 (significant quality gain).
          For INT8, simple RTN is often sufficient — the error is so small
          that GPTQ's expensive optimisation adds negligible benefit.

❌ WRONG: You need to retrain the model for quantization
✓ RIGHT:  Post-Training Quantization (PTQ) with calibration data works
          excellently for INT4+ (GPTQ, AWQ). Only for extreme compression
          (INT2-INT3) or if you want maximum quality should you consider
          Quantization-Aware Training (QAT).
```

---

## 8. Exercises

1. **Symmetric INT4**: Quantize the vector [-0.5, 0.2, -1.0, 0.8, 0.3] to INT4 symmetric (range [-8, 7]). Compute scale, quantized values, dequantized values, and max error.

2. **Memory Savings**: For LLaMA-3 70B with G=128 group quantization: compute total memory for INT4 weights + FP16 scales. Compare to BF16 baseline. What's the effective bits per parameter?

3. **NF4 Construction**: Given a Gaussian N(0,1) distribution, compute the first 4 NF4 quantization bin boundaries by finding the 4-quantile points of the CDF. What are the bin midpoints?

4. **SmoothQuant**: An activation vector is [0.1, 0.2, 50.0, -0.1, 60.0]. With INT8 quantization (range [-128,127]): compute the quantized values and errors. Now apply smoothing with s=[1,1,10,1,10] and re-quantize. Compare errors.
