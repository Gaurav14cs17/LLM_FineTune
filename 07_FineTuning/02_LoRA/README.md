# Understanding LoRA: Low-Rank Adaptation

*A comprehensive guide to efficient fine-tuning from mathematical foundations to implementation*

---

**LoRA (Low-Rank Adaptation)** allows fine-tuning a 8B-parameter LLM on a single consumer GPU by training only a tiny fraction of parameters. Instead of updating all 8 billion weights, LoRA adds small "adapter" matrices and trains only those.

This guide explains the mathematical foundation, why it works, and how to choose the right hyperparameters.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 The Problem: Full Fine-Tuning is Expensive](#11-the-problem-full-fine-tuning-is-expensive)
   - [1.2 The LoRA Idea](#12-the-lora-idea)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Intrinsic Dimensionality](#2-intrinsic-dimensionality)
   - [2.1 The Theoretical Foundation](#21-the-theoretical-foundation)
3. [LoRA Reparameterisation](#3-lora-reparameterisation)
   - [3.1 The Weight Decomposition](#31-the-weight-decomposition)
   - [3.2 Forward Pass](#32-forward-pass)
   - [3.3 The α/r Scaling](#33-the-αr-scaling)
4. [Initialisation](#4-initialisation)
   - [4.1 Why B=0, A~N?](#41-why-b0-an)
   - [4.2 Gradient Flow Proof](#42-gradient-flow-proof)
5. [Rank Selection](#5-rank-selection)
   - [5.1 SVD Theory](#51-svd-theory)
   - [5.2 Practical Guidelines](#52-practical-guidelines)
6. [Memory and Parameter Savings](#6-memory-and-parameter-savings)
   - [6.1 Exact Calculations for LLaMA-3 8B](#61-exact-calculations-for-llama-3-8b)
7. [Summary](#7-summary)
   - [7.1 All Formulas Quick Reference](#71-all-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 The Problem: Full Fine-Tuning is Expensive

```
┌─────────────────────────────────────────────────────────────┐
│  MEMORY COST of Full Fine-Tuning LLaMA-3 8B                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Component                     Memory (GB)                  │
│  ─────────────────────────────────────────────────────      │
│  BF16 model weights              16 GB                      │
│  BF16 gradients                  16 GB                      │
│  FP32 master weights             32 GB                      │
│  FP32 Adam 1st moment (m)        32 GB                      │
│  FP32 Adam 2nd moment (v)        32 GB                      │
│  ─────────────────────────────────────────────────────      │
│  TOTAL                          128 GB  ← needs 2× A100!   │
│                                                             │
│  Consumer GPU (RTX 4090): 24 GB  ← cannot fit!             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The LoRA Idea

```
┌─────────────────────────────────────────────────────────────┐
│  INTUITION: Residuals are Low-Rank                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Full fine-tuning updates: W₀ + ΔW                         │
│                                                             │
│  LoRA hypothesis: ΔW has LOW INTRINSIC RANK                 │
│                                                             │
│  Instead of storing ΔW ∈ ℝᵈˣᵏ (d×k = 16.8M parameters),   │
│  decompose: ΔW = B × A   where rank(B×A) ≤ r               │
│                                                             │
│  B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ  with r << min(d,k)                 │
│                                                             │
│  Parameters: r×(d+k)  instead of  d×k                      │
│  For d=k=4096, r=16:  16×8192 = 131K  vs  4096²=16.8M     │
│  SAVINGS: 128× fewer parameters for this matrix!           │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Instead of reprinting a whole book to fix a chapter, you write a small "errata sheet" listing only the changes. LoRA is the errata sheet for a pretrained model.

### 1.3 Pipeline Summary

```
INPUT: pretrained model W₀, fine-tuning dataset D
        ↓
Step 1: Freeze W₀  (no gradients flow through W₀)
        ↓
Step 2: Add low-rank adapters B, A to each targeted layer
        Effective weight: W = W₀ + (α/r) × B × A
        ↓
Step 3: Train ONLY B and A on dataset D
        ↓
Step 4 (optional): Merge W = W₀ + (α/r)BA at inference
        (zero latency overhead after merging!)
        ↓
OUTPUT: fine-tuned model with minimal compute and memory
```

---

## 2. Intrinsic Dimensionality

### 2.1 The Theoretical Foundation

```
┌─────────────────────────────────────────────────────────────┐
│  AGHAJANYAN ET AL. (2021): Key Finding                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  "The fine-tuning update Δθ lives in a                      │
│   low-dimensional subspace of parameter space."            │
│                                                             │
│  For RoBERTa (125M parameters):                             │
│    Fine-tuning in FULL space (125M dims): ✓ (baseline)     │
│    Fine-tuning in 200-dim subspace:       ~90% of baseline  │
│    Fine-tuning in 100-dim subspace:       ~85% of baseline  │
│                                                             │
│  d_int ≈ 200 << 125,000,000 ← enormous compression!        │
│                                                             │
│  VISUALISATION:                                             │
│  Full parameter space ℝ^{125M}:                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │   θ_pretrained ●                                    │   │
│  │                  ╲ fine-tuning trajectory           │   │
│  │                   ╲ (lives in ~200-dim subspace)    │   │
│  │                    ● θ_SFT                          │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  WHY? Pre-training has already structured weight space.    │
│  Fine-tuning only needs to STEER the model, not teach it.  │
│  The "steering directions" are low-dimensional.            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. LoRA Reparameterisation

### 3.1 The Weight Decomposition

For a weight matrix W₀ ∈ ℝᵈˣᵏ (frozen):

```
FULL FINE-TUNING:
  W = W₀ + ΔW        ΔW ∈ ℝᵈˣᵏ  (d×k parameters to train)

LORA DECOMPOSITION:
  W = W₀ + ΔW = W₀ + B × A

  where:
    B ∈ ℝᵈˣʳ   (d×r trainable parameters)
    A ∈ ℝʳˣᵏ   (r×k trainable parameters)
    r << min(d, k)   (the "rank" of the adapter)

RANK CONSTRAINT:
  rank(B×A) ≤ min(rank(B), rank(A)) ≤ r

  This is the key: the update ΔW is constrained to have rank ≤ r.
  The update lives in a rank-r subspace of the full update space.
```

### 3.2 Forward Pass

```
ORIGINAL:   h = x × W₀              (frozen)
LORA:       h = x × W₀  +  x × B × A × (α/r)
                ↑ frozen      ↑ trainable LoRA path

STEP BY STEP:
  1. x ∈ ℝᵈ           (input token embedding)
  2. x × W₀ ∈ ℝᵏ      (frozen base computation)
  3. x × A ∈ ℝʳ        (project to rank-r space)  ← small!
  4. (x×A) × B ∈ ℝᵏ    (project back to output space)
  5. Sum: h = xW₀ + xAB×(α/r)

COMPUTATIONAL COST vs FULL:
  Full W = W₀ + ΔW: same as original (one d×k matmul)
  LoRA:  x×W₀ (d×k) + x×A (d×r) + ×B (r×k)
  
  Extra FLOPs: 2×d×r + 2×r×k = 2r(d+k)
  Relative:    2r(d+k) / 2dk = (d+k)/(dk/r) ≈ r/d for d≈k
  For d=4096, r=16: extra ≈ 16/4096 = 0.4% overhead — negligible!
```

### 3.3 The α/r Scaling

```
┌─────────────────────────────────────────────────────────────┐
│  WHY SCALE BY α/r?                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  The update magnitude is: ‖BA‖ ∝ r (grows with rank r)     │
│                                                             │
│  If you increase r without α:  ‖ΔW‖ grows → need to        │
│  manually re-tune the learning rate for every r value.      │
│                                                             │
│  With α/r scaling:                                          │
│  ‖ΔW‖ = ‖BA‖ × α/r ≈ const × α/r × r = const × α         │
│                                                             │
│  → The effective update scale depends only on α, NOT r!    │
│  → You can change r without re-tuning the learning rate.   │
│                                                             │
│  COMMON CHOICES:                                            │
│    α = r   → effective LR = η (same as global LR)          │
│    α = 2r  → effective LR = 2η (2× global LR)             │
│    α = 1   → effective LR = η/r (decreases with r)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Initialisation

### 4.1 Why B=0, A~N?

```
REQUIREMENT: ΔW = B×A = 0 at the START of training.

Why? The pretrained model already works.
If ΔW ≠ 0 at start, we immediately disrupt the model before any learning.

OPTIONS:
  Option A: A = 0, B ~ N(0, σ²)
    → ΔW = B×0 = 0 ✓
    → But: gradient ∂L/∂A = Bᵀ × (∂L/∂Z) = 0 at start (since B≠0 but ∂L/∂A = Bᵀ × grad)
    Actually: ∂L/∂A = Bᵀ × ∂L/∂Z ← Bᵀ is random non-zero → A DOES get gradient!
    Wait, this means A gets gradient. Let's check Option B too.

  Option B (STANDARD): B = 0, A ~ N(0, σ²)
    → ΔW = 0×A = 0  ✓
    → Gradient ∂L/∂B = (∂L/∂Z) × Aᵀ ≠ 0 since A is random non-zero ✓
    → B immediately starts updating from step 1!
```

### 4.2 Gradient Flow Proof

```
Let Z = B×A, output h = x×W₀ + x×Z×(α/r)

∂L/∂B = ∂L/∂Z × Aᵀ       (matrix chain rule)
∂L/∂A = Bᵀ × ∂L/∂Z       (matrix chain rule)

AT INITIALISATION (B=0, A random):

  ∂L/∂B = ∂L/∂Z × Aᵀ ≠ 0   (A is random non-zero → Aᵀ ≠ 0) ✓
  ∂L/∂A = 0ᵀ × ∂L/∂Z = 0   ✗ (A gets NO gradient initially)

AFTER FIRST STEP:
  B moves from 0 to B₁ = 0 − η × (∂L/∂B) ≠ 0
  Now ∂L/∂A = B₁ᵀ × ∂L/∂Z ≠ 0 → A starts learning too ✓

RESULT:
  ┌─────────────────────────────────────────────────────────┐
  │  Step 0: ΔW = 0  (no disruption to pretrained model)   │
  │  Step 1: B moves, ΔW = B₁×A ≠ 0  (learning begins)    │
  │  Step 2: Both B and A update  (full learning)           │
  └─────────────────────────────────────────────────────────┘
  
  The initialisation B=0, A~N creates a "warm start" for B.
```

---

## 5. Rank Selection

### 5.1 SVD Theory

```
┌─────────────────────────────────────────────────────────────┐
│  THEOREM: LoRA with r ≥ r* can exactly represent            │
│           any update ΔW* with rank(ΔW*) = r*.               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PROOF:                                                     │
│  By SVD: ΔW* = U × Σ × Vᵀ                                  │
│          U ∈ ℝᵈˣʳ*, Σ ∈ ℝʳ*ˣʳ* (diagonal), V ∈ ℝᵏˣʳ*     │
│                                                             │
│  Set B = U × Σ^{1/2} ∈ ℝᵈˣʳ*                              │
│      A = Σ^{1/2} × Vᵀ ∈ ℝʳ*ˣᵏ                             │
│                                                             │
│  Then B×A = UΣ^{1/2} × Σ^{1/2}Vᵀ = U×Σ×Vᵀ = ΔW* ✓  ■    │
│                                                             │
│  → LoRA with r ≥ r* can express ANY rank-r* update         │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Practical Guidelines

```
WHAT r CAPTURES (by singular value energy):

  SVD of true ΔW*: singular values σ₁ ≥ σ₂ ≥ σ₃ ≥ ...

  Energy in top r singular values:
  E(r) = Σᵢ₌₁ʳ σᵢ² / Σᵢ σᵢ²

  ┌──────────────────────────────────────────────────────────┐
  │  Typical LLM fine-tuning singular value decay:          │
  │                                                          │
  │  σ₁ = 10.0    σ₂ = 5.0    σ₃ = 2.5    σ₄ = 1.2    ... │
  │                                                          │
  │  E(1)  = 100/156 = 64%                                  │
  │  E(4)  = (100+25+6.25+1.44)/156 = 85%                  │
  │  E(16) ≈ 98%   ← captures most of the update           │
  │  E(64) ≈ 99.5%                                          │
  └──────────────────────────────────────────────────────────┘

EMPIRICAL QUALITY vs RANK:
  r = 1:    ~85% quality vs full FT  (very aggressive compression)
  r = 4:    ~95% quality
  r = 16:   ~99% quality   ← sweet spot for most tasks
  r = 64:   ~99.5% quality
  r = 256:  ~99.9% quality  (diminishing returns)

TASK-BASED RECOMMENDATIONS:
  Task type              Recommended r  Notes
  ─────────────────────────────────────────────────────────
  Style / format         r = 4 – 8     minimal adaptation
  Instruction following  r = 8 – 16    most common use case
  Knowledge-intensive    r = 32 – 64   medical, legal, code
  Near-full fine-tuning  r = 128 – 256 approaching full FT
```

---

## 6. Memory and Parameter Savings

### 6.1 Exact Calculations for LLaMA-3 8B

**LoRA adapter parameter count (r=16, targeting all attention + FFN):**

```
Per layer calculations:

  Matrix          Dimensions          LoRA params (r=16)
  ────────────────────────────────────────────────────────
  q_proj          4096×4096           16×(4096+4096) = 131,072
  k_proj          4096×1024           16×(4096+1024) = 81,920
  v_proj          4096×1024           16×(4096+1024) = 81,920
  o_proj          4096×4096           16×(4096+4096) = 131,072
  gate_proj       4096×14336          16×(4096+14336) = 294,912
  up_proj         4096×14336          16×(4096+14336) = 294,912
  down_proj       14336×4096          16×(14336+4096) = 294,912
  ────────────────────────────────────────────────────────
  Per layer total                     1,310,720  ≈ 1.3M
  × 32 layers                         41,943,040 ≈ 42M trainable
```

**Memory comparison:**

```
┌─────────────────────────────────────────────────────────────┐
│  MEMORY COMPARISON                                          │
├─────────────────────────────────────────────────────────────┤
│                              Full FT    LoRA (r=16)        │
│  ────────────────────────────────────────────────────────  │
│  Model weights (BF16)         16 GB       16 GB (frozen)   │
│  Gradients (BF16)             16 GB        0.17 GB  ✓      │
│  FP32 master weights          32 GB        0.34 GB  ✓      │
│  Adam 1st moment (FP32)       32 GB        0.34 GB  ✓      │
│  Adam 2nd moment (FP32)       32 GB        0.34 GB  ✓      │
│  ────────────────────────────────────────────────────────  │
│  TOTAL                       128 GB       17.2 GB          │
│                                                             │
│  SAVINGS: 7.4×  ← fits in 2× RTX 4090 (48 GB total)!     │
│                                                             │
│  With QLoRA (4-bit base model): further reduces to ~6 GB  │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Summary

### 7.1 All Formulas Quick Reference

**Weight Decomposition:**

```
W = W₀ + ΔW = W₀ + B × A × (α/r)
  W₀ ∈ ℝᵈˣᵏ  frozen,  B ∈ ℝᵈˣʳ,  A ∈ ℝʳˣᵏ  trainable
```

**Forward Pass:**

```
h = x × W₀  +  (x × A × B) × (α/r)
    ↑ frozen      ↑ LoRA path (tiny compute overhead)
```

**Parameter count:**

```
Full:  d × k
LoRA:  r × (d + k)
Ratio: r(d+k) / dk  ≈  r/d for d≈k  (e.g., 16/4096 = 0.4%)
```

**Gradient at init (B=0, A~N):**

```
∂L/∂B = ∂L/∂Z × Aᵀ ≠ 0   (B gets gradient from step 1)
∂L/∂A = Bᵀ × ∂L/∂Z = 0   (A waits for step 2)
```

| Hyperparameter | Typical value | Effect |
|----------------|---------------|--------|
| r (rank) | 8 – 64 | Higher r = more capacity, more memory |
| α (scale numerator) | = r or 2r | α/r sets effective learning rate |
| Target matrices | q,v (minimal) or q,k,v,o,ffn | More targets = more capacity |
| LR | 1e-4 to 3e-4 | Higher than full FT (smaller param count) |

### 7.2 Common Mistakes

```
❌ WRONG: Initialising both A=0 and B=0
✓ RIGHT:  B=0, A~N(0,σ²) — one must be random for gradients to flow

❌ WRONG: Forgetting the α/r scaling factor
✓ RIGHT:  Always divide by r: ΔW = BA × (α/r). Without it, changing r
          changes the effective update magnitude and breaks LR tuning.

❌ WRONG: Only applying LoRA to q_proj and v_proj and expecting full FT quality
✓ RIGHT:  For knowledge-intensive tasks, target all 7 matrices per layer.
          q,v only is fine for style/format tasks.

❌ WRONG: Using r=256 for a simple formatting task
✓ RIGHT:  Start small (r=8), increase if quality is insufficient.
          More parameters = more overfitting risk on small datasets.

❌ WRONG: Merging weights before evaluating multiple tasks
✓ RIGHT:  Keep the adapter separate. Merge only for final deployment.
          Multi-task: use different adapters, hot-swap at inference.
```

---

## 8. Exercises

1. **Parameter Savings**: For d=2048, k=2048, r=16, compute the parameter savings ratio over full fine-tuning. How many times fewer parameters does LoRA use?

2. **Gradient Flow**: Prove that if B=0, A~N, then ∂L/∂B ≠ 0 at step 0 but ∂L/∂A = 0. Work through the matrix derivatives explicitly.

3. **Rank vs Quality**: If the true update ΔW* has singular values [10, 4, 1.5, 0.5, 0.2, 0.1, ...] (geometric decay with ratio 0.4), compute E(r) for r=1,2,4,8. What rank captures 95% of the update energy?

4. **Memory Budget**: For a fine-tuning run with r=32 targeting all 7 matrices in 32 layers of LLaMA-3 8B: compute the total trainable parameters and the total memory for FP32 Adam states. Compare to full fine-tuning memory.
