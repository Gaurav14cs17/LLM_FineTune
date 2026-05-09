# 04 — Mixed Precision Training

## 1. Floating Point Number Systems — Full Analysis

```
IEEE 754 floating point format:  sign | exponent | mantissa
                                  1 bit   E bits     M bits

VALUE represented:  (−1)^sign × 2^{exponent−bias} × (1 + mantissa/2^M)

FORMATS USED IN LLM TRAINING:
                 Sign  Exp  Mantissa  Total  Range          Precision
  FP32 (full)     1     8     23       32    ±3.4×10³⁸      ~7 decimal digits
  FP16 (half)     1     5     10       16    ±65504          ~3 decimal digits
  BF16 (brain)    1     8      7       16    ±3.4×10³⁸      ~2 decimal digits
  FP8 E4M3        1     4      3        8    ±448            ~1 decimal digit
  FP8 E5M2        1     5      2        8    ±57344          ~0.5 decimal digit

KEY INSIGHT — BF16 vs FP16:
  BF16 truncates MANTISSA (fewer significant digits)
  FP16 truncates EXPONENT (smaller range)
  
  BF16 range = FP32 range (same exponent!) → no overflow on large activations
  FP16 can overflow above 65504 — common during training!
  
  ┌─────────────────────────────────────────────────────────────┐
  │  FP32: ██████████████████████████████████  (full precision) │
  │  BF16: ████████████████                    (half precision) │
  │         same exponent range ✓   fewer mantissa bits         │
  │                                                             │
  │  FP16: ████████████████                    (half precision) │
  │         narrower range!      more mantissa bits             │
  └─────────────────────────────────────────────────────────────┘
```

## 2. Mixed Precision Training — Mathematical Framework

```
MIXED PRECISION STRATEGY (Micikevicius et al. 2018):
  
  Store: master weights in FP32  (high precision for accumulation)
  Compute: forward/backward in BF16 (fast, uses Tensor Cores)
  
  FORWARD PASS:
    W_bf16 = cast(W_fp32, BF16)                ← on-the-fly, no storage
    a_l = f_l(a_{l-1}, W_bf16)                 ← BF16 matmul
    Store activations a_l in BF16 (gradient checkpointing: some in FP32)
  
  BACKWARD PASS:
    Compute gradients g_bf16 in BF16
    g_fp32 = cast(g_bf16, FP32)                ← upcast before accumulation
  
  OPTIMIZER STEP (all in FP32):
    m_t = β₁ m_{t-1} + (1-β₁) g_fp32
    v_t = β₂ v_{t-1} + (1-β₂) g_fp32²
    W_fp32 -= η × m̂_t / (√v̂_t + ε)

WHY FP32 FOR OPTIMIZER STATES?
  AdamW maintains per-parameter running statistics m, v.
  These accumulate SMALL increments over many steps.
  
  BF16 mantissa precision: ~0.8% relative error (1/2^7 = 0.0078)
  For a weight W = 0.1 and gradient update Δ = 0.0001:
    In BF16: 0.1 + 0.0001 = 0.1001 rounds to 0.1   ← update LOST!
    In FP32: 0.1 + 0.0001 = 0.1001                 ← preserved ✓
  
  Over 100K training steps, accumulated BF16 rounding errors destroy convergence.
```

## 3. Loss Scaling — Why FP16 Needs It

```
FP16 UNDERFLOW PROBLEM:
  FP16 smallest positive normal: ~6.1×10⁻⁵
  FP16 smallest positive subnormal: ~5.96×10⁻⁸
  
  Gradient magnitudes in deep LLMs: often in range [10⁻⁷, 10⁻⁴]
  → Many gradients UNDERFLOW to 0 in FP16!
  
  ┌─────────────────────────────────────────────────────────────┐
  │  TRUE gradient distribution:                                │
  │                    ██                                       │
  │                  ██████                                     │
  │                ██████████                                   │
  │              ████████████████                               │
  │  ──────────────────────────────────────────────────────    │
  │  10⁻⁸  10⁻⁶  10⁻⁴  10⁻²   1                              │
  │        ↑ FP16 underflow below ~6×10⁻⁵                      │
  │        These gradients → 0 in FP16 → silent accuracy loss  │
  └─────────────────────────────────────────────────────────────┘

DYNAMIC LOSS SCALING solution:
  1. Multiply loss by scale factor S (e.g., S=2048)
     loss_scaled = S × loss
  2. Backward pass computes S × gradients  (larger → no underflow!)
  3. Divide by S before optimizer update:  g_true = g_scaled / S
  4. Check for overflow (Inf/NaN in g_scaled):
       If Inf/NaN found:
         Discard the entire step (don't update weights)
         S ← S / 2  (halve the scale)
       Else (no overflow for K=2000 consecutive steps):
         S ← S × 2  (double the scale)

WHY BF16 DOESN'T NEED LOSS SCALING:
  BF16 smallest normal: ~9.2×10⁻⁴¹  (same as FP32!)
  → Gradients never underflow in BF16
  → No loss scaling needed → simpler training code!
```

## 4. Memory Breakdown — Exact Calculations

```
For 8B parameter model (e.g. LLaMA-3 8B):

INFERENCE (BF16):
  Model weights:  8×10⁹ × 2 bytes = 16 GB
  KV cache (varies with context, see Ch.8)

TRAINING (BF16 forward/backward + FP32 optimizer):
  Component                    Formula           Size
  ─────────────────────────────────────────────────────
  BF16 model weights           8B × 2B    =  16 GB
  BF16 gradients               8B × 2B    =  16 GB
  FP32 master weights          8B × 4B    =  32 GB  ← for update accumulation
  FP32 Adam 1st moment m       8B × 4B    =  32 GB
  FP32 Adam 2nd moment v       8B × 4B    =  32 GB
  ─────────────────────────────────────────────────────
  Parameters/gradients total              = 128 GB
  Activations (batch=1, L=4096) ≈          30 GB  ← depends on seq length
  ─────────────────────────────────────────────────────
  TOTAL ≈ 158 GB  ← needs 2× A100 80GB at minimum!

WITH GRADIENT CHECKPOINTING:
  Store only activations at layer boundaries, recompute within each layer.
  Activation memory: ~8 GB  (saving 22 GB)
  Cost: +33% FLOPs (recompute forward for each layer during backward)

8-BIT OPTIMIZER (bitsandbytes):
  m, v stored in INT8 instead of FP32: 8B × 1B × 2 = 16 GB  (saves 48 GB!)
  Total with INT8 optim + gradient checkpointing: ≈ 76 GB → fits on single A100!
```

## 5. Tensor Core Utilisation

```
NVIDIA Tensor Cores: hardware units that compute A×B+C in BF16/FP16 in one cycle.
  Peak throughput on A100: 312 TFLOPS (BF16) vs 19.5 TFLOPS (FP32)
  → BF16 is 16× faster for matmuls!

BUT: only works on specific matrix dimensions (multiples of 8 or 16).

ALIGNMENT REQUIREMENTS (for optimal Tensor Core use):
  d, d_ff, |V| should all be multiples of 64 (ideally 128 or 256).
  LLaMA-3 8B:
    d = 4096 = 32 × 128 ✓
    d_ff = 14336 = 112 × 128 ✓
    |V| = 128000 = 1000 × 128 ✓

EFFECTIVE FLOPS utilisation (MFU — Model FLOPs Utilisation):
  MFU = (actual FLOPs/sec) / (peak Tensor Core FLOPs/sec)
  
  Typical LLM training MFU: 40–50% on A100
  (rest lost to memory bandwidth, communication overhead, etc.)
  FlashAttention improves MFU by ~10–15 percentage points.
```

---

## Exercises

1. Compute the total training memory (in GB) for LLaMA-3 8B with BF16 activations, FP32 master weights, FP32 optimizer states, and activation checkpointing (assume 2 GB saved activations).
2. Why does BF16 not need loss scaling but FP16 does? Use the exponent bit counts to explain the underflow threshold difference.
3. A gradient value is 2×10⁻⁸. In FP16 (min ≈6×10⁻⁵), does it underflow? In BF16 (min ≈9×10⁻⁴¹)? In FP32 (min ≈1.2×10⁻³⁸)?
