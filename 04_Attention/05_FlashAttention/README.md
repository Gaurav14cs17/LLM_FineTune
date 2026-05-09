# Understanding FlashAttention

*IO-aware exact attention in O(n) memory — tiling, online softmax, and the memory hierarchy*

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 The Memory Bottleneck](#11-the-memory-bottleneck)
   - [1.2 Key Insight: Tiling](#12-key-insight-tiling)
2. [GPU Memory Hierarchy](#2-gpu-memory-hierarchy)
   - [2.1 HBM vs SRAM](#21-hbm-vs-sram)
   - [2.2 IO Complexity of Standard Attention](#22-io-complexity-of-standard-attention)
3. [The FlashAttention Algorithm](#3-the-flashattention-algorithm)
   - [3.1 Online Softmax (the key mathematical trick)](#31-online-softmax-the-key-mathematical-trick)
   - [3.2 Tiled Forward Pass](#32-tiled-forward-pass)
   - [3.3 Pseudocode](#33-pseudocode)
4. [Memory and Compute Analysis](#4-memory-and-compute-analysis)
   - [4.1 Memory: O(n) not O(n²)](#41-memory-on-not-on2)
   - [4.2 IO Complexity](#42-io-complexity)
5. [Numerical Example](#5-numerical-example)
6. [Common Mistakes](#6-common-mistakes)
7. [Exercises](#7-exercises)

---

## 1. Overview

### 1.1 The Memory Bottleneck

```
┌─────────────────────────────────────────────────────────────┐
│  STANDARD ATTENTION writes a FULL n×n matrix to HBM:        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: S = Q × Kᵀ     ∈ ℝ^{n×n}  → WRITE to HBM (n² floats)│
│  Step 2: P = softmax(S)  ∈ ℝ^{n×n}  → WRITE to HBM (n² floats)│
│  Step 3: O = P × V       ∈ ℝ^{n×d}  → WRITE to HBM (n×d floats)│
│                                                             │
│  Total HBM traffic: 3×n² + n×d ≈ 3n² (dominated by n²)    │
│                                                             │
│  For n=32K, d=128:                                          │
│    S matrix: 32768² × 2 bytes = 2.15 GB                    │
│    P matrix: 32768² × 2 bytes = 2.15 GB                    │
│    Total: ~4.3 GB written to slow HBM PER LAYER PER HEAD!  │
│                                                             │
│  At 2 TB/s HBM bandwidth: 4.3 GB / 2 TB/s = 2.15 ms/head  │
│  × 32 heads × 32 layers = 2.2 seconds just for IO!         │
│                                                             │
│  FlashAttention: NEVER writes S or P to HBM → O(n) memory  │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: Standard attention is like computing a spreadsheet formula cell-by-cell and saving the entire intermediate spreadsheet to disk each time. FlashAttention computes everything in RAM and only saves the final answer to disk.

### 1.2 Key Insight: Tiling

```
INSTEAD OF: compute full n×n attention matrix at once
DO:         process Q, K, V in TILES that fit in fast SRAM

  ┌─────────────────────────────────────────────────────┐
  │  Q (n×d)  split into blocks of size B_q × d         │
  │  K (n×d)  split into blocks of size B_k × d         │
  │  V (n×d)  split into blocks of size B_k × d         │
  │                                                     │
  │  For each Q-block (B_q rows of Q):                   │
  │    For each K,V-block (B_k rows of K,V):             │
  │      - Load Q_block, K_block, V_block into SRAM     │
  │      - Compute partial attention in SRAM (fast!)    │
  │      - Update running output O using online softmax │
  │    End                                               │
  │  End                                                 │
  │                                                     │
  │  SRAM usage: B_q×d + 2×B_k×d + B_q×B_k ≈ O(B²+Bd) │
  │  HBM writes: ONLY the final O (n×d) — no S, no P!  │
  └─────────────────────────────────────────────────────┘
```

---

## 2. GPU Memory Hierarchy

### 2.1 HBM vs SRAM

```
┌──────────────────────────────────────────────────────────────┐
│  A100 GPU MEMORY HIERARCHY:                                  │
├──────────────────┬────────────┬──────────────────────────────┤
│  Level           │ Size       │ Bandwidth                    │
├──────────────────┼────────────┼──────────────────────────────┤
│  Registers       │ ~256 KB    │ ~∞ (instantaneous)           │
│  SRAM (shared)   │ 20 MB      │ 19 TB/s                      │
│  HBM (VRAM)      │ 80 GB      │ 2 TB/s                       │
│  CPU DRAM        │ ~1 TB      │ 0.05 TB/s                    │
├──────────────────┴────────────┴──────────────────────────────┤
│                                                              │
│  SRAM is 9.5× FASTER than HBM but 4000× SMALLER!            │
│  → Keep hot data in SRAM, minimise HBM round-trips.         │
│                                                              │
│  FlashAttention MAXIMISES SRAM usage:                        │
│    - Q, K, V tiles loaded once into SRAM                    │
│    - All attention computation happens in SRAM              │
│    - Only final output O is written to HBM                  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 IO Complexity of Standard Attention

```
STANDARD ATTENTION IO (read + write from/to HBM):

  READ:   Q (n×d) + K (n×d) + V (n×d) = 3nd
  WRITE:  S (n×n) + P (n×n) + O (n×d) = 2n² + nd
  READ:   S (n×n) for softmax + P (n×n) for P×V = 2n²
  
  TOTAL IO: Θ(n² + nd) = Θ(n²)  (since n >> d typically)

FLASHATTENTION IO:
  READ:   Q (n×d) + K (n×d) + V (n×d) = 3nd
  WRITE:  O (n×d) = nd
  
  TOTAL IO: Θ(nd)  ← NO n² term!
  
  SPEEDUP: n²/nd = n/d
  For n=32K, d=128: speedup ≈ 256× less IO!
```

---

## 3. The FlashAttention Algorithm

### 3.1 Online Softmax (the key mathematical trick)

```
PROBLEM: softmax requires knowing ALL scores before normalising.
  softmax(sᵢ) = exp(sᵢ) / Σⱼ exp(sⱼ)
  → Need the full row sum Σⱼ exp(sⱼ) to compute any single output!

SOLUTION: ONLINE SOFTMAX — update incrementally as new blocks arrive.

  Maintain running statistics per query:
    m(current) = max score seen so far (for numerical stability)
    l(current) = sum of exp(scores - m) seen so far
    O(current) = running weighted output

  When processing a NEW block of K scores [s_new]:
    m_new = max(m_old, max(s_new))         ← update max
    
    l_new = l_old × exp(m_old − m_new)     ← rescale old sum
          + Σ exp(s_new − m_new)            ← add new contributions
    
    O_new = O_old × (l_old/l_new) × exp(m_old − m_new)  ← rescale old output
          + (1/l_new) × Σ exp(s_new − m_new) × V_new    ← add new output

  After ALL blocks processed: O = correct attention output!
  
  KEY: at no point do we store the full n×n matrix.
  We only ever store O(B_k) scores for the current tile.
```

### 3.2 Tiled Forward Pass

```
┌──────────────────────────────────────────────────────────────┐
│  FLASHATTENTION FORWARD (one attention head):                │
│                                                              │
│  Input: Q, K, V ∈ ℝ^{n×d} in HBM                           │
│  Output: O ∈ ℝ^{n×d} in HBM                                 │
│                                                              │
│  Block sizes: B_q = ⌈SRAM / (4d)⌉,  B_k = same             │
│                                                              │
│  Initialise: O = 0,  m = -∞,  l = 0  (in HBM, per query)   │
│                                                              │
│  For j = 1 to ⌈n/B_k⌉:      ← iterate over K,V blocks     │
│    Load K_j, V_j from HBM to SRAM                           │
│                                                              │
│    For i = 1 to ⌈n/B_q⌉:    ← iterate over Q blocks        │
│      Load Q_i, O_i, m_i, l_i from HBM to SRAM              │
│                                                              │
│      S_ij = Q_i × K_jᵀ ∈ ℝ^{B_q × B_k}  (in SRAM!)        │
│                                                              │
│      m_new = max(m_i, rowmax(S_ij))                          │
│      P_ij = exp(S_ij − m_new)      (stable softmax)         │
│      l_new = l_i × exp(m_i − m_new) + rowsum(P_ij)          │
│      O_new = O_i × (l_i/l_new)×exp(m_i−m_new) + P_ij×V_j/l_new│
│                                                              │
│      Write O_new, m_new, l_new back to HBM                  │
│    End                                                       │
│  End                                                         │
│                                                              │
│  Return O (in HBM) — EXACT same result as standard attention!│
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Pseudocode

```python
# FlashAttention Forward (simplified)
def flash_attention(Q, K, V, B_q, B_k):
    n, d = Q.shape
    O = zeros(n, d)
    m = full(n, -inf)  # running max per query
    l = zeros(n)       # running sum per query
    
    for j in range(0, n, B_k):           # K,V block loop
        Kj = K[j:j+B_k]                  # load K block to SRAM
        Vj = V[j:j+B_k]                  # load V block to SRAM
        
        for i in range(0, n, B_q):       # Q block loop
            Qi = Q[i:i+B_q]              # load Q block to SRAM
            
            Sij = Qi @ Kj.T / sqrt(d)    # B_q × B_k scores (in SRAM)
            
            m_new = maximum(m[i:i+B_q], Sij.max(axis=1))
            exp_old = exp(m[i:i+B_q] - m_new)
            Pij = exp(Sij - m_new[:, None])
            l_new = l[i:i+B_q] * exp_old + Pij.sum(axis=1)
            
            O[i:i+B_q] = O[i:i+B_q] * (l[i:i+B_q] * exp_old / l_new)[:, None]
            O[i:i+B_q] += (Pij @ Vj) / l_new[:, None]
            
            m[i:i+B_q] = m_new
            l[i:i+B_q] = l_new
    
    return O  # mathematically IDENTICAL to standard attention
```

---

## 4. Memory and Compute Analysis

### 4.1 Memory: O(n) not O(n²)

```
MEMORY COMPARISON:

  Standard attention:
    S ∈ ℝ^{n×n}:  n² × sizeof(dtype)  ← the O(n²) culprit!
    P ∈ ℝ^{n×n}:  n² × sizeof(dtype)
    O ∈ ℝ^{n×d}:  n×d × sizeof(dtype)
    Total: Θ(n²)
    
    For n=128K, BF16: 128K² × 2 = 32 GB — IMPOSSIBLE on one GPU!

  FlashAttention:
    Q, K, V, O ∈ ℝ^{n×d}:  4×n×d × sizeof(dtype)
    m, l ∈ ℝ^n:             2×n × sizeof(dtype)
    SRAM tile:              O(B²) — reused, not stored
    Total: Θ(n×d)
    
    For n=128K, d=128, BF16: 4 × 128K × 128 × 2 = 128 MB ← EASY!
```

### 4.2 IO Complexity

```
FLASHATTENTION IO (total HBM reads + writes):

  Reads: Q once fully = nd
         K, V: each read ⌈n/B_q⌉ times = n²d / SRAM_size
  Writes: O written ⌈n/B_k⌉ times, final = nd

  Total IO = Θ(n²d² / SRAM_size)

  With SRAM=20MB, n=32K, d=128, BF16:
    Standard: 2n² × 2 = 4 GB
    Flash:    n²×d / SRAM ≈ 32K²×128×2 / 20M ≈ 13 GB... 
    BUT practical: flash is 2-4× faster due to COMPUTE utilisation!

WHY FLASH IS FASTER DESPITE SAME FLOPs:
  1. Fewer HBM round-trips (kernel fusion — one kernel, not 3)
  2. Better GPU occupancy (tiles keep all SMs busy)
  3. No intermediate n² matrix allocation (saves memory bandwidth)
  
  PRACTICAL SPEEDUP: 2-4× for training, 3-5× for inference
  PRACTICAL MEMORY SAVINGS: O(n) instead of O(n²) → enables 16× longer contexts!
```

---

## 5. Numerical Example

```
EXAMPLE: n=4, d=2, B_q=B_k=2 (tiny for illustration)

  Q = [[1,0],[0,1],[1,1],[0,0]]    K = [[1,0],[0,1],[1,0],[0,1]]
  V = [[1,2],[3,4],[5,6],[7,8]]    scale = 1/√2

  STANDARD (full attention matrix):
    S = Q×Kᵀ/√2 = [[0.71,0,0.71,0],[0,0.71,0,0.71],[0.71,0.71,0.71,0.71],[0,0,0,0]]
    P = softmax(S)  (row-wise)
    O = P × V

  FLASHATTENTION (tiled, B=2):
    Processing Q_block=[Q₀,Q₁] with K_block=[K₀,K₁]:
      S_partial = [[0.71,0],[0,0.71]]
      m = [0.71, 0.71], l = [exp(0.71)+exp(0), exp(0)+exp(0.71)]
      O_partial = softmax(S_partial) × V[0:2]
    
    Then with K_block=[K₂,K₃]:
      S_partial = [[0.71,0],[0,0.71]]
      UPDATE m, l, O using online softmax formulas...
      
    FINAL O = same as standard (EXACT, not approximate!)
```

---

## 6. Common Mistakes

```
❌ WRONG: FlashAttention is an approximation (like sparse attention)
✓ RIGHT:  FlashAttention computes EXACT standard attention.
          It's an IO-efficient implementation, NOT an algorithm change.
          The output is bit-for-bit identical to standard attention.

❌ WRONG: FlashAttention reduces FLOPs (fewer multiplications)
✓ RIGHT:  FlashAttention has the SAME number of FLOPs as standard attention.
          Both compute n² dot products. The speedup comes from better
          memory access patterns (less HBM IO), not fewer computations.

❌ WRONG: FlashAttention only helps for long sequences
✓ RIGHT:  FlashAttention helps at ALL sequence lengths because of kernel
          fusion (fewer kernel launches, better GPU occupancy).
          But the MEMORY savings are most dramatic for long sequences.

❌ WRONG: FlashAttention cannot do causal masking
✓ RIGHT:  FlashAttention supports causal masking by simply skipping tiles
          where all mask values are -∞ (upper triangle in causal).
          This also SAVES compute: ~50% of tiles are skipped for causal!
```

---

## 7. Exercises

1. **Memory Comparison**: For n=64K, d=128, 32 heads, BF16: compute total attention memory for (a) standard attention (b) FlashAttention. What is the savings ratio?

2. **Tile Size**: A100 has 20 MB SRAM shared across streaming multiprocessors. For d=128, BF16: compute the maximum block size B such that Q_block (B×d) + K_block (B×d) + V_block (B×d) + S_tile (B×B) fit in SRAM.

3. **IO Analysis**: For n=32K, d=128, B=256: how many times is each K,V block loaded from HBM? Compare total IO bytes to standard attention.

4. **Online Softmax**: Given scores in two blocks: block 1 = [2, 1] and block 2 = [3, 0]. Compute softmax incrementally using the online formula (m, l update). Verify the result matches standard softmax over [2, 1, 3, 0].
