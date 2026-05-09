# 05 — FlashAttention

## 1. The Memory Hierarchy Problem

```
GPU Memory Hierarchy:
  ┌─────────────────────────────────────────────────────────┐
  │  REGISTERS     ~256 KB     ~∞ TB/s bandwidth            │  ← fastest, tiny
  ├─────────────────────────────────────────────────────────┤
  │  SRAM (L1/L2)  ~20 MB      ~19 TB/s bandwidth  (A100)   │  ← fast, small
  ├─────────────────────────────────────────────────────────┤
  │  HBM (VRAM)    ~80 GB      ~2 TB/s bandwidth   (A100)   │  ← slow, large
  └─────────────────────────────────────────────────────────┘

STANDARD ATTENTION I/O (L=32K, d=128, h=32):
  HBM writes:  S = QKᵀ  →  L² floats  = 32768² × 2B = 2 GB  ← write to HBM
  HBM reads:   A = softmax(S)          = 2 GB              ← read back from HBM
  HBM writes:  O = A·V                 = 2 GB              ← write back to HBM
  Total HBM I/O: ~6 GB per attention call

  At 2 TB/s bandwidth:  6 GB / 2 TB/s = 3 ms per call
  × 32 layers × many steps = BOTTLENECK!
```

## 2. FlashAttention Key Insight: Tiled Computation

```
IDEA: Compute attention in tiles that fit in SRAM.
Never write the full L×L attention matrix to HBM.

SRAM budget: ~20 MB
Block size: B_r × B_c × d × sizeof(float) ≤ SRAM/2

For d=128, B_r=B_c=64:  64×64×128×2B = 1 MB < 20 MB ✓

TILING LAYOUT:
  Q blocks:  ┌────┬────┬────┐
             │ Q₁ │ Q₂ │ Q₃ │   (each Qᵢ block has B_r rows)
             └────┴────┴────┘

  K,V blocks: ┌────┬────┬────┐
              │ K₁ │ K₂ │ K₃ │   (each Kⱼ block has B_c columns)
              └────┴────┴────┘

  For each Q block Qᵢ:
    For each K,V block (Kⱼ, Vⱼ):
      Compute Sᵢⱼ = Qᵢ × Kⱼᵀ / √d_k   (in SRAM)
      Update running softmax stats
      Update output Oᵢ                 (in SRAM)
    Write Oᵢ to HBM once               (ONE write per Q block)
```

## 3. Online Softmax — The Key Algorithm

```
PROBLEM: softmax(row i) requires knowing ALL scores in that row.
         But we process tiles — we don't see all scores at once.

SOLUTION: Online softmax maintains running statistics:

  For each tile processed:
  ┌────────────────────────────────────────────────────────────┐
  │  State: m = running max, ℓ = running normaliser            │
  │         O = running output                                  │
  │                                                            │
  │  New tile has scores s_new:                                │
  │  m_new = max(m, max(s_new))                               │
  │  ℓ_new = ℓ × exp(m − m_new)  +  ∑ exp(s_new − m_new)     │
  │  O_new = O × ℓ/ℓ_new × exp(m−m_new)                      │
  │        + (1/ℓ_new) × exp(s_new − m_new) × V_new          │
  └────────────────────────────────────────────────────────────┘

This is numerically stable AND incremental — only requires SRAM!

CORRECTNESS: At the end, O = (1/ℓ) × ∑ exp(s − m) × V = standard softmax output ✓
```

## 4. Memory Savings Visualised

```
STANDARD ATTENTION (L=32K):
  Allocated in HBM:
  ┌──────────────────────────────────────────────────────────┐
  │  S matrix: L×L = 32K×32K×2B = 2 GB  ← stored in HBM!   │
  │  A matrix: L×L = 2 GB               ← stored in HBM!    │
  │  O matrix: L×d = 32K×128×2B = 8 MB  ← stored in HBM    │
  └──────────────────────────────────────────────────────────┘
  Peak memory: ~4 GB per attention layer

FLASHATTENTION (L=32K):
  ┌──────────────────────────────────────────────────────────┐
  │  SRAM: tiles of Q,K,V,S,O (each ≤ 1 MB)                │
  │  HBM:  only O (final output = 8 MB)                     │
  │  S, A matrices: NEVER written to HBM!                   │
  └──────────────────────────────────────────────────────────┘
  Peak memory: ~8 MB per attention layer  (250× reduction!)
```

## 5. Speed Comparison

```
                Standard    FlashAttention-1  FlashAttention-2
FLOPs:          O(L²d)      O(L²d)            O(L²d)           (same math!)
HBM reads:      O(L²+Ld)    O(L²d/M)          O(L²d/M)         (M=SRAM size)
Memory:         O(L²)       O(L)              O(L)
A100 speedup:   1×          ~3×               ~4.5×
H100 (FA-3):    1×          ~3×               ~6×

Wall-clock time (A100, L=2048, d=64, h=32):
  Standard:        12 ms
  FlashAttention:   3 ms   (4× faster!)
  
Training throughput improvement (GPT-3 scale):
  Standard:   40K tokens/sec
  FA-2:      140K tokens/sec  (3.5× improvement)
```

---

## Exercises

1. For L=8192, d=128, h=32, BF16: compute the memory savings of FlashAttention vs standard attention.
2. Why does FlashAttention have the same FLOPs but fewer HBM I/O operations?
3. Sketch the online softmax update. Show that the final output is identical to standard softmax.
