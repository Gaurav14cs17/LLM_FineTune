# Understanding Mixture of Experts (MoE) for LLMs

*A comprehensive guide to sparse expert routing — from gating mathematics to load balancing*

---

**Mixture of Experts (MoE)** is the architecture behind Mixtral 8×7B, GPT-4, DeepSeek-V2, and Switch Transformer. Instead of using one giant FFN at every layer, MoE routes each token to a *subset* of experts, achieving the performance of a massive model with the inference cost of a much smaller one.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 The Scaling Dilemma](#11-the-scaling-dilemma)
   - [1.2 Sparse vs Dense](#12-sparse-vs-dense)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [The MoE Layer](#2-the-moe-layer)
   - [2.1 Architecture](#21-architecture)
   - [2.2 Router / Gating Function](#22-router--gating-function)
   - [2.3 Top-K Expert Selection](#23-top-k-expert-selection)
3. [Load Balancing](#3-load-balancing)
   - [3.1 The Collapse Problem](#31-the-collapse-problem)
   - [3.2 Auxiliary Loss Derivation](#32-auxiliary-loss-derivation)
   - [3.3 Expert Capacity Factor](#33-expert-capacity-factor)
4. [Training Dynamics](#4-training-dynamics)
   - [4.1 Gradient Flow Through the Router](#41-gradient-flow-through-the-router)
   - [4.2 Router Z-Loss](#42-router-z-loss)
5. [Parameter vs Active Parameters](#5-parameter-vs-active-parameters)
   - [5.1 Mixtral 8×7B Numbers](#51-mixtral-87b-numbers)
   - [5.2 FLOPs Analysis](#52-flops-analysis)
6. [Advanced: Expert Parallelism](#6-advanced-expert-parallelism)
7. [Summary](#7-summary)
   - [7.1 Formulas Quick Reference](#71-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 The Scaling Dilemma

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: Bigger model = better quality BUT more FLOPs      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Dense 70B model:                                           │
│    Quality:  Excellent                                      │
│    FLOPs/token: 2×70B = 140 GFLOPs                         │
│    Inference:  EXPENSIVE                                     │
│                                                             │
│  Dense 7B model:                                            │
│    Quality:  Good                                           │
│    FLOPs/token: 2×7B = 14 GFLOPs                           │
│    Inference:  CHEAP                                         │
│                                                             │
│  MoE 8×7B (Mixtral) — THE BEST OF BOTH:                    │
│    Total params:  46.7B (8 experts of ~7B each + shared)    │
│    Active params: 12.9B (only 2 experts used per token)     │
│    Quality:  MATCHES 70B dense                              │
│    FLOPs/token: 2×12.9B = 25.8 GFLOPs  ← 5.4× cheaper!   │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Instead of one professor who knows everything (expensive consultant), you have 8 specialists. For each question, you route it to the 2 most relevant experts. You get world-class answers at a fraction of the cost.

### 1.2 Sparse vs Dense

| Property | Dense (LLaMA) | MoE (Mixtral) |
|----------|--------------|---------------|
| Architecture | Every token uses ALL params | Each token uses subset |
| Total params | N | E × N (much larger) |
| Active params per token | N | Top-K/E × N |
| Quality at fixed compute | Baseline | Higher |
| Memory (weights) | N | E × N (more!) |
| Memory (KV cache) | Same | Same |
| Training data needed | D | ~4× D (more params need more data) |

### 1.3 Pipeline Summary

```
INPUT: token hidden state x ∈ ℝᵈ (from attention output)
        ↓
Step 1: ROUTER computes gating scores for all E experts
        g(x) = softmax(W_g × x)  → probabilities over E experts
        ↓
Step 2: SELECT Top-K experts (typically K=2)
        expert_ids = argtop-K(g(x))
        ↓
Step 3: COMPUTE outputs from selected experts only
        y_i = Expert_i(x)   for i ∈ expert_ids
        ↓
Step 4: COMBINE with gating weights (normalised)
        y = Σᵢ ĝᵢ(x) × Expert_i(x)
        ↓
OUTPUT: y ∈ ℝᵈ  (same shape as input — drop-in replacement for FFN)
```

---

## 2. The MoE Layer

### 2.1 Architecture

```
STANDARD TRANSFORMER BLOCK:
  y = x + Attention(RMSNorm(x))
  y = y + FFN(RMSNorm(y))          ← this FFN is replaced by MoE

MoE TRANSFORMER BLOCK:
  y = x + Attention(RMSNorm(x))
  y = y + MoE(RMSNorm(y))          ← MoE replaces FFN

INSIDE MoE:
  ┌───────────────────────────────────────────────────────────┐
  │                                                           │
  │  Input x ─────────┬──────────────────────────────────→ +  │
  │                    │                                   ↑  │
  │                    ↓                                   │  │
  │              ┌──────────┐                              │  │
  │              │  Router   │  g = softmax(W_g × x)       │  │
  │              └──────────┘                              │  │
  │                    │  pick top-K                        │  │
  │         ┌──────────┼──────────┐                        │  │
  │         ↓          ↓          ↓                        │  │
  │   ┌──────────┐ ┌──────────┐ ┌──────────┐              │  │
  │   │Expert 1  │ │Expert 2  │ │Expert 3  │ ...E experts │  │
  │   │(SwiGLU)  │ │(SwiGLU)  │ │(SwiGLU)  │              │  │
  │   └──────────┘ └──────────┘ └──────────┘              │  │
  │         │          │          │                        │  │
  │         ↓ g₁       ↓ g₂      ↓ (not selected)         │  │
  │         ×          ×                                   │  │
  │         └──────────┴───────────────────────────────────┘  │
  │                                                           │
  └───────────────────────────────────────────────────────────┘
  
  Each Expert_i is an independent FFN (same architecture as dense FFN):
    Expert_i(x) = SwiGLU(x × W_gate_i, x × W_up_i) × W_down_i
```

### 2.2 Router / Gating Function

```
ROUTER (learned linear layer + softmax):

  h = W_g × x          W_g ∈ ℝᴱˣᵈ,  x ∈ ℝᵈ  →  h ∈ ℝᴱ
  g(x) = softmax(h)    g ∈ ℝᴱ,  Σᵢ gᵢ = 1

VARIABLES:
  E = number of experts (8 in Mixtral)
  d = model hidden dimension (4096)
  W_g = router weight matrix (8 × 4096 = 32,768 params per MoE layer)

EXAMPLE (E=8, showing router outputs for one token):
  h = W_g × x = [2.1, -0.3, 1.8, 0.1, -1.2, 0.5, -0.8, 1.5]
  g = softmax(h) = [0.31, 0.03, 0.23, 0.04, 0.01, 0.06, 0.02, 0.17]
                              ↑                                    ↑
  Top-2:  Expert 0 (g=0.31) and Expert 2 (g=0.23)
```

### 2.3 Top-K Expert Selection

```
TOP-K GATING (Mixtral uses K=2):

Step 1: Select indices of top-K largest gates
  indices = argtop-K(g(x)) = {i₁, i₂, ..., iₖ}

Step 2: Zero out non-selected gates
  ĝᵢ = gᵢ  if i ∈ indices
  ĝᵢ = 0   otherwise

Step 3: Renormalise selected gates to sum to 1
  ĝ_norm_i = ĝᵢ / Σⱼ∈indices ĝⱼ

Step 4: Final output = weighted sum of expert outputs
  y = Σᵢ∈indices  ĝ_norm_i × Expert_i(x)

NUMERICAL EXAMPLE:
  g = [0.31, 0.03, 0.23, 0.04, 0.01, 0.06, 0.02, 0.17]
  Top-2 indices: {0, 2}
  
  ĝ₀ = 0.31,  ĝ₂ = 0.23   (others = 0)
  Normalise: ĝ₀ = 0.31/(0.31+0.23) = 0.574
             ĝ₂ = 0.23/(0.31+0.23) = 0.426
  
  Output: y = 0.574 × Expert₀(x) + 0.426 × Expert₂(x)

SPARSITY:
  Only 2/8 = 25% of expert parameters are activated per token!
  FLOPs = 2/8 × Full-MoE-FFN-FLOPs ≈ cost of one ~2× wide FFN
```

---

## 3. Load Balancing

### 3.1 The Collapse Problem

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM: WITHOUT BALANCING, ROUTING COLLAPSES               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Token distribution across experts (collapsed):             │
│  Expert 0: ████████████████████████████████ 80%             │
│  Expert 1: ████                             10%             │
│  Expert 2: ██                                5%             │
│  Expert 3: █                                 3%             │
│  Expert 4: ░                                 1%             │
│  Expert 5: ░                                 0.5%           │
│  Expert 6: ░                                 0.3%           │
│  Expert 7: ░                                 0.2%           │
│                                                             │
│  WHY IT HAPPENS:                                            │
│  1. Expert 0 gets slightly more tokens initially (random)   │
│  2. Expert 0 gets more gradient signal → improves faster    │
│  3. Router sends even MORE tokens to Expert 0 (it's better)│
│  4. Other experts get starved → never improve               │
│  5. Positive feedback loop → total collapse to 1-2 experts │
│                                                             │
│  RESULT: 6/8 experts are DEAD. Model capacity ≈ 2× dense,  │
│  not 8× as intended. Massive parameter waste!              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Auxiliary Loss Derivation

```
GOAL: Encourage uniform distribution of tokens across experts.

DEFINE (for a batch of T tokens):
  fᵢ = (1/T) × Σₜ 1[expert i is in top-K for token t]
      = fraction of tokens routed to expert i

  Pᵢ = (1/T) × Σₜ gᵢ(xₜ)
      = average gate probability for expert i

IDEAL: fᵢ = 1/E and Pᵢ = 1/E for all i (perfectly uniform)

AUXILIARY LOAD-BALANCING LOSS:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  L_aux = α × E × Σᵢ₌₁ᴱ fᵢ × Pᵢ                        │
  │                                                          │
  │  α = auxiliary loss coefficient (typically 0.01)         │
  │  E = number of experts                                   │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

WHY THIS WORKS (mathematical intuition):
  If all fᵢ = 1/E and Pᵢ = 1/E:
    L_aux = α × E × Σᵢ (1/E)(1/E) = α × E × E × 1/E² = α  (minimum)

  If Expert 0 gets everything: f₀=1, P₀=1:
    L_aux = α × E × (1×1 + 0×0 + ...) = α × E  (E× larger = penalised!)

  The product fᵢ × Pᵢ penalises experts that are both:
  - POPULAR (high fᵢ — many tokens routed)
  - CONFIDENT (high Pᵢ — router strongly prefers them)
  This breaks the positive feedback loop that causes collapse.

TOTAL LOSS:
  L_total = L_NLL + L_aux
           = −(1/T)Σₜ log P(wₜ|w<t) + α × E × Σᵢ fᵢ × Pᵢ
```

### 3.3 Expert Capacity Factor

```
CAPACITY FACTOR (Switch Transformer):

  max_tokens_per_expert = capacity_factor × (T / E)

  capacity_factor = 1.0: each expert gets exactly its fair share
  capacity_factor = 1.25: 25% overflow buffer allowed

WHAT HAPPENS WHEN EXPERT IS FULL:
  If expert i already has max_tokens_per_expert tokens:
    New tokens intended for expert i are DROPPED
    (passed through with 0 weight, or routed to next-best expert)

  ┌─────────────────────────────────────────────────────────┐
  │  capacity_factor = 1.0:  strict load balancing            │
  │    + All experts used equally                             │
  │    - Some tokens dropped → quality loss                   │
  │                                                           │
  │  capacity_factor = ∞:    no limit                         │
  │    + No tokens dropped                                    │
  │    - Potential collapse / uneven compute                   │
  │                                                           │
  │  capacity_factor = 1.25: practical balance (Mixtral)     │
  │    + Rarely drops tokens                                  │
  │    + Limits worst-case load imbalance                     │
  └─────────────────────────────────────────────────────────┘
```

---

## 4. Training Dynamics

### 4.1 Gradient Flow Through the Router

```
THE GATING WEIGHT ĝᵢ(x) is differentiable w.r.t. router params W_g:

  y = Σᵢ ĝᵢ(x) × Expert_i(x)
  
  ∂L/∂W_g = Σᵢ (∂L/∂y) × Expert_i(x) × (∂ĝᵢ/∂W_g)
  
  The router learns WHICH expert to route each token to,
  by seeing which expert produces lower loss.
  
  INTUITION: If Expert₂ gives a bad answer for "math tokens":
    → Loss is high → gradient tells router "don't send math to Expert₂"
    → Router learns to route math tokens to Expert₀ instead

PROBLEM: the top-K selection (argmax) is NOT differentiable!
  Solution 1: Straight-through estimator (gradient passes through unchanged)
  Solution 2: Only differentiate through the gate WEIGHTS ĝᵢ, not the selection
  
  Mixtral uses Solution 2: the selected expert indices are fixed (no gradient),
  but the WEIGHTS assigned to selected experts are differentiable.
```

### 4.2 Router Z-Loss

```
ADDITIONAL STABILISATION (ST-MoE, PaLM):

  L_z = (1/T) × Σₜ (log Σᵢ exp(hᵢ(xₜ)))²

  This penalises LARGE router logits (before softmax).
  
  WHY:
  If router logits are [50, -50, ...] → softmax ≈ [1, 0, ...]
  → very sharp routing → unstable (tiny perturbation flips routing)
  → numerical overflow risk in exp(50)
  
  Z-loss keeps logits moderate → routing is smoother and more stable.
  Coefficient: typically 0.001 × main loss.
```

---

## 5. Parameter vs Active Parameters

### 5.1 Mixtral 8×7B Numbers

```
MIXTRAL 8×7B ARCHITECTURE:
  Layers: 32
  d = 4096
  h = 32 attention heads
  d_k = 128
  Experts: E = 8 per MoE layer
  Top-K: K = 2
  FFN intermediate: 14,336 per expert

PARAMETER COUNT:
  ┌───────────────────────────────────────────────────────────┐
  │  Component               Per layer    × 32 layers         │
  │  ─────────────────────────────────────────────────────    │
  │  Attention (shared):     33.6M        1.07B               │
  │  Expert FFNs (8×):       8 × 57.7M    14.77B             │
  │  Router:                 32K          1M                  │
  │  RMSNorm:               8K           256K                │
  │  ─────────────────────────────────────────────────────    │
  │  Total per layer:        495M                             │
  │  Total model:            ~46.7B parameters                │
  │                                                           │
  │  ACTIVE per token:       12.9B  (attention + 2 experts)  │
  │  Ratio: 12.9/46.7 = 27.6% of params active per token    │
  └───────────────────────────────────────────────────────────┘
```

### 5.2 FLOPs Analysis

```
FLOPs per token (forward pass):

DENSE 7B (LLaMA-2 7B):
  Attention: 2 × 4096 × 4096 × 4 = 134M   (Q,K,V,O projections)
  FFN:       2 × 4096 × 11008 × 3 = 270M   (gate, up, down)
  Per layer: ~404M FLOPs
  Total:     32 × 404M = 12.9 GFLOPs

MIXTRAL 8×7B (Top-2):
  Attention: 2 × 4096 × 4096 × 4 = 134M   (shared, same as dense)
  FFN (2 experts): 2 × (2 × 4096 × 14336 × 3) = 703M  (2/8 experts)
  Per layer: ~837M FLOPs
  Total:     32 × 837M = 26.8 GFLOPs  ← ~2× dense 7B

COMPARISON:
  ┌────────────────────────────────────────────────────────────┐
  │  Model          Total Params  Active Params  FLOPs/token  │
  │  LLaMA-2 7B     7B            7B             14 GFLOPs    │
  │  Mixtral 8×7B   46.7B         12.9B          26 GFLOPs    │
  │  LLaMA-2 70B    70B           70B            140 GFLOPs   │
  │                                                            │
  │  Mixtral matches LLaMA-2 70B quality at 5.4× fewer FLOPs! │
  └────────────────────────────────────────────────────────────┘
```

---

## 6. Advanced: Expert Parallelism

```
┌─────────────────────────────────────────────────────────────┐
│  DISTRIBUTING MoE ACROSS GPUs                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STANDARD (Tensor Parallelism): split EACH expert across GPUs│
│  MoE (Expert Parallelism): put DIFFERENT experts on GPUs   │
│                                                             │
│  GPU 0: Expert 0, Expert 1                                  │
│  GPU 1: Expert 2, Expert 3                                  │
│  GPU 2: Expert 4, Expert 5                                  │
│  GPU 3: Expert 6, Expert 7                                  │
│                                                             │
│  ALL-TO-ALL COMMUNICATION:                                  │
│  Step 1: Each GPU runs router → determines which token      │
│          goes to which expert (on which GPU)                │
│  Step 2: All-to-all: send tokens to their assigned GPU     │
│  Step 3: Each GPU processes tokens with its local experts  │
│  Step 4: All-to-all: send results back to originating GPU  │
│                                                             │
│  ┌─────┐    all-to-all    ┌─────┐    all-to-all    ┌─────┐│
│  │Route│ ──────────────►  │Comp │ ──────────────►  │Merge││
│  │(all)│                  │(local)│                 │(all)││
│  └─────┘                  └─────┘                  └─────┘│
│                                                             │
│  BANDWIDTH REQUIREMENT:                                     │
│  Each token: 2 × d × sizeof(dtype) bytes                   │
│  T tokens × 2 × 4096 × 2 bytes = T × 16 KB per step       │
│  High-bandwidth interconnect (NVLink) essential!            │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Summary

### 7.1 Formulas Quick Reference

**Router/Gating:**

```
g(x) = softmax(W_g × x)     W_g ∈ ℝᴱˣᵈ
indices = argtop-K(g(x))
ĝᵢ = gᵢ / Σⱼ∈indices gⱼ    (normalised top-K)
```

**MoE output:**

```
y = Σᵢ∈indices ĝᵢ(x) × Expert_i(x)
```

**Load-balancing loss:**

```
L_aux = α × E × Σᵢ fᵢ × Pᵢ
  fᵢ = fraction of tokens routed to expert i
  Pᵢ = average gate probability for expert i
```

**Router Z-loss:**

```
L_z = (1/T) × Σₜ (log Σᵢ exp(hᵢₜ))²
```

| Hyperparameter | Typical value | Effect |
|----------------|---------------|--------|
| E (experts) | 8 | More experts = more capacity, more memory |
| K (top-K) | 2 | Higher K = more compute, better quality |
| α (aux loss) | 0.01 | Too high → forced balance, too low → collapse |
| Capacity factor | 1.25 | Too low → drops tokens, too high → imbalance |

### 7.2 Common Mistakes

```
❌ WRONG: MoE uses 8× more FLOPs than a dense model of same active params
✓ RIGHT:  Only K experts are active per token. FLOPs ≈ K/E × full MoE.
          For K=2, E=8: only 25% of expert FLOPs are used.

❌ WRONG: MoE requires 8× less GPU memory than total param count suggests
✓ RIGHT:  ALL expert weights must be in memory (or loaded on demand).
          Mixtral 8×7B needs memory for 46.7B params, even though only 12.9B
          are active. MoE saves FLOPs, NOT memory for weights.

❌ WRONG: Experts specialise in different "topics" (math expert, code expert...)
✓ RIGHT:  Analysis shows experts don't cleanly specialise by topic.
          They specialise by SYNTAX/PATTERN more than semantics.
          Some experts handle punctuation, others handle rare tokens, etc.

❌ WRONG: You can simply convert a dense model to MoE by duplicating the FFN
✓ RIGHT:  Initialising all experts as copies WORKS for starting training,
          but expert specialisation requires significant additional training.
          "Upcycling" (Komatsuzaki et al.) trains for ~100B additional tokens.
```

---

## 8. Exercises

1. **FLOPs Calculation**: For E=8, K=2, d=4096, FFN intermediate=14336: compute FLOPs for one MoE layer (including attention). Compare to a dense layer with the same FFN width.

2. **Load Balancing**: A batch of 1024 tokens with E=8 has routing: Expert 0 gets 400 tokens, Expert 1 gets 300, others get ~50 each. Compute fᵢ for each expert. What is L_aux with α=0.01? Compare to the ideal uniform case.

3. **Memory Budget**: Mixtral 8×7B in BF16: compute total weight memory. If you have 2× A100 (80 GB each), can you fit the model? What about KV cache for 4096 context?

4. **Expert Parallelism**: For E=8, batch=512, d=4096, BF16: compute the all-to-all communication volume per MoE layer. If NVLink bandwidth is 600 GB/s, what is the communication time?
