# 03 — QLoRA: Quantised Low-Rank Adaptation

## 1. Quantisation Background — NF4 Data Type

```
QUANTISATION: represent floating-point values with fewer bits.

4-BIT QUANTISATION (16× compression vs FP32):
  Original: FP32 value w ∈ ℝ
  Quantised: q ∈ {0,1,...,15}  (4 bits = 16 discrete levels)
  
  LINEAR QUANTISATION:
    q = round(w / scale_factor)   where scale = (w_max − w_min) / 15
    w ≈ q × scale_factor + zero_point
  
  PROBLEM: LLM weights follow a roughly Gaussian N(0,1) distribution.
    Linear quantisation wastes levels on the flat tails!
    Most values near 0 (dense region) get poor precision.

NF4 (Normal Float 4, Dettmers et al. 2023):
  Designed specifically for Gaussian-distributed weights!
  
  CONSTRUCTION: choose 16 quantisation levels to be the expected values of
  16 equal-probability bins of N(0,1):
  
  qi = E[X | X ∈ [Φ⁻¹((i)/16), Φ⁻¹((i+1)/16)]]   for i = 0, ..., 15
  
  where Φ⁻¹ is the inverse CDF of N(0,1).
  
  NF4 LEVELS (symmetric, normalised to [-1, 1]):
  {-1.000, -0.694, -0.520, -0.394, -0.289, -0.194, -0.105, -0.021,
    0.079,  0.187,  0.303,  0.432,  0.584,  0.776,  1.000, [unused]}
  
  Wait — that's only 15 values. NF4 uses 16 values symmetric around 0.
  
  INFORMATION THEORY ARGUMENT:
    NF4 is INFORMATION-THEORETICALLY OPTIMAL for normally distributed weights.
    
    PROOF: Lloyd's optimal quantisation theorem states that for any distribution,
    the quantisation error is minimised when levels are at the centroids of each
    equal-probability bin (equal-frequency quantisation).
    
    For N(0,1): equal-probability bins = equal percentile ranges.
    NF4 uses these exact centroids → minimises MSE for Gaussian weights!
```

## 2. Double Quantisation — Memory Calculation

```
STANDARD QUANTISATION:
  For each block of 64 weights, we need one quantisation constant (scale factor).
  Extra memory: (N_params / 64) × 4 bytes per scale  (FP32 scale)
  
  For 8B parameters:
    = 8×10⁹/64 × 4 bytes = 500 MB of quantisation constants!  (non-trivial)

DOUBLE QUANTISATION (QLoRA):
  Quantise the quantisation constants themselves!
  
  Step 1: Quantise model weights W to NF4 with FP32 scale factors
    Memory for scales: (N/64) × 4 bytes = 500 MB (as above)
  
  Step 2: Group scale factors into blocks of 256, quantise scales to FP8
    Memory for scale-of-scales: (N/64)/256 × 1 byte = 1.95 MB (tiny!)
    Memory for FP8 scales:      (N/64) × 1 byte = 125 MB
  
  TOTAL MEMORY FOR QUANTISATION CONSTANTS:
    Without DQ: 500 MB
    With DQ:    125 MB + 1.95 MB ≈ 127 MB  (saves 373 MB!)

TOTAL MEMORY — 4-BIT MODEL:
  Weights (NF4, 4 bits per param): 8×10⁹ × 0.5 bytes = 4 GB
  Quantisation constants (DQ):     127 MB
  ─────────────────────────────────────────
  Total quantised model: ~4.1 GB  (vs 16 GB for BF16!)
  
  MEMORY SAVINGS: 16 GB → 4.1 GB ← fits in single consumer GPU (12 GB VRAM)!
```

## 3. Paged Optimisers — Dealing with GPU OOM

```
PROBLEM: Even with 4-bit base model, the LoRA adapter optimiser states need memory.
  LoRA r=16: 42M trainable params × 8 bytes per param (FP32 Adam m,v) = 336 MB

SPIKES during training:
  Gradient accumulation: temporary copies of full activations.
  For batch with very long sequences (e.g., 4096 tokens × BS=4):
    Activation memory peak ≈ 10-30 GB!  → GPU OOM even with 4-bit model.

PAGED OPTIMISER SOLUTION (using NVIDIA unified memory):
  1. Allocate optimiser states as PAGE-ABLE CPU memory.
  2. Page in relevant pages to GPU when needed for update.
  3. Page out to CPU when GPU memory is needed for activations.
  
  KEY MECHANISM:
    - NVIDIA's CUDA Unified Memory automatically handles CPU↔GPU page transfers.
    - During forward/backward: optimiser states offloaded to CPU.
    - During optimizer step: page needed param group's states to GPU, update, page out.
  
  OVERHEAD:
    PCIe bandwidth: 16 GB/s (gen4) → transfer 336 MB ≈ 21 ms overhead
    Training step ≈ 200 ms → overhead ≈ 10%  (acceptable!)
  
  BENEFIT: can train on GPUs with 12-16 GB VRAM instead of needing 40-80 GB!
```

## 4. QLoRA Forward Pass — Dequantisation

```
INFERENCE/TRAINING with QLoRA:

BASE MODEL (frozen, stored in NF4):
  W_NF4 ∈ {level_0, ..., level_15}^{d_out × d_in/2}   ← 4 bits per element

FORWARD PASS for input x:
  Step 1: DEQUANTISE block-by-block:
    For each block b of 64 weights:
      W_fp16[b] = lookup_NF4(W_NF4[b]) × scale_b_fp16
      (NF4 lookup table: 16 values → 16 BF16 values)
  
  Step 2: COMPUTE base output (BF16):
    h_base = x · W_fp16   (standard BF16 matmul)
  
  Step 3: COMPUTE LoRA output (BF16):
    h_lora = (x · A) · B × (α/r)   (LoRA adapter, both A and B in BF16)
  
  Step 4: SUM:
    h = h_base + h_lora   (final output)

GRADIENT FLOW:
  ∂L/∂A and ∂L/∂B flow normally through step 3 (trainable).
  ∂L/∂W_NF4 = 0  (base weights are frozen, no gradient needed).
  
  This is why NF4 storage is fine: we only ever READ the base weights,
  never UPDATE them. The quantisation error is fixed (no gradient noise).

QUANTISATION ERROR BOUND:
  For NF4 on N(0,σ²) distributed weights:
  E[|w - q(w)|²] ≤ σ²/4  ← error bounded by σ²/16 × 4 = σ²/4  per element
  
  For σ = 0.02 (LLaMA init): MSE ≤ (0.02)²/4 = 10⁻⁴ per element.
  Relative error: √(10⁻⁴)/(0.02) = 50% ← sounds bad!
  
  BUT: the LoRA adapter can compensate for systematic quantisation error!
  The adapter learns to correct the bias introduced by quantisation.
  This is why QLoRA quality ≈ LoRA quality ≈ full fine-tuning quality.
```

## 5. Memory Budget Summary

```
FINE-TUNING LLaMA-3 8B with QLoRA (r=16, all attention + FFN):

Component                    Bytes per param   Total
────────────────────────────────────────────────────
Base model (NF4)              0.5 bytes         4.0 GB
Quantisation constants (DQ)   ~0.016 bytes      128 MB
LoRA A, B params (BF16)       2 bytes/lora      84 MB  (42M × 2)
Gradient for LoRA (BF16)      2 bytes           84 MB
Adam m (FP32)                 4 bytes          168 MB
Adam v (FP32)                 4 bytes          168 MB
Activations (batch=1, L=512)  variable         ~1 GB (with checkpointing)
────────────────────────────────────────────────────
TOTAL ≈ 5.7 GB   ← fits comfortably on RTX 3090 (24 GB), even 4090!

COMPARISON:
  Method        │ Min VRAM   │ Quality vs full FT
  ──────────────┼────────────┼────────────────────
  Full FT (BF16)│ 160 GB     │ 100% (baseline)
  LoRA (BF16)   │ 18 GB      │ ~99%
  QLoRA (NF4)   │ 6-8 GB     │ ~98.5%  (marginal loss)
  LoRA (INT8)   │ 12 GB      │ ~98%
```

---

## Exercises

1. Derive the NF4 quantisation levels for the first and last levels (i=0 and i=15). Use the formula qi = E[X | X ∈ [Φ⁻¹(i/16), Φ⁻¹((i+1)/16)]] and the N(0,1) distribution.
2. Compute the exact memory savings from double quantisation for a 70B parameter model. How much is saved compared to using FP32 quantisation constants?
3. Estimate the maximum batch size that fits in 24 GB VRAM for QLoRA fine-tuning of LLaMA-3 8B with sequence length 2048. Account for all components in the memory budget table above.
