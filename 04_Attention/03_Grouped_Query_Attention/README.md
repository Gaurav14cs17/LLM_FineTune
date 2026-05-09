# 03 — Grouped Query Attention (GQA) & Multi-Query Attention (MQA)

## 1. The KV Cache Memory Problem — Exact Derivation

```
During inference, we cache K and V tensors for all past tokens.
Each new token needs all previous K,V to compute attention.

KV CACHE SIZE for MHA:
  Per layer, per token:
    K: h × d_k × bytes = h × (d/h) × bytes = d × bytes
    V: h × d_v × bytes = d × bytes
    Total: 2d × bytes per token per layer

  For LLaMA-3 8B (d=4096, L=32 layers, BF16=2 bytes):
    Per token: 2 × 4096 × 2 × 32 = 524,288 bytes = 512 KB
  
  For a context of C tokens:
    Total KV cache = 512 KB × C
    
    C = 4096:   2.1 GB
    C = 8192:   4.2 GB
    C = 32768: 16.8 GB  ← eats most of A100's 80 GB!

  For batch size B=16 with 4K context:
    KV cache = 16 × 2.1 GB = 33.6 GB  (> 40% of total GPU memory)
  
  ← This severely limits throughput!
```

## 2. MQA — Mathematical Formulation

```
MULTI-QUERY ATTENTION (Shazeer 2019):
  Use h query projections but ONLY 1 key and 1 value projection.

  W_Q^i ∈ ℝ^{d×d_k}   for i = 1, ..., h   (h different)
  W_K   ∈ ℝ^{d×d_k}                         (shared, ONE matrix)
  W_V   ∈ ℝ^{d×d_v}                         (shared, ONE matrix)

  head_i = Attention( X·W_Q^i,  X·W_K,  X·W_V )
  MQA(X) = Concat(head₁,...,headₕ) · W_O

KV CACHE REDUCTION:
  MHA KV:  2 × h × d_k = 2d parameters per token per layer
  MQA KV:  2 × 1 × d_k = 2d/h  parameters per token per layer
  
  REDUCTION FACTOR = h   (32× reduction for h=32!)

QUALITY TRADE-OFF:
  Each head sees the SAME K,V → heads can only differ in their Q projection.
  This limits the diversity of attention patterns.
  
  Empirically: MQA loses ~2-5% on benchmarks vs MHA (Ainslie et al. 2023).
```

## 3. GQA — Mathematical Formulation

```
GROUPED QUERY ATTENTION (Ainslie et al. 2023):
  G groups, each group has h/G query heads sharing ONE K,V pair.

  For group g (g = 1, ..., G):
    W_K^g ∈ ℝ^{d×d_k}    (one K projection per group)
    W_V^g ∈ ℝ^{d×d_v}    (one V projection per group)
    W_Q^{g,i} for i=1,...,h/G  (h/G query projections per group)

  head_{g,i} = Attention( X·W_Q^{g,i},  X·W_K^g,  X·W_V^g )

  GQA(X) = Concat(head_{1,1}, ..., head_{G,h/G}) · W_O

KV CACHE CALCULATION:
  Per token per layer:
    K: G × d_k × 2 bytes
    V: G × d_k × 2 bytes
    Total: 2G × d_k × 2 bytes = 4G × d_k bytes

  For LLaMA-3 8B (G=8, d_k=128, 32 layers, BF16):
    Per token: 4 × 8 × 128 × 32 = 131,072 bytes = 128 KB
    
    MHA would be: 4 × 32 × 128 × 32 = 524,288 bytes = 512 KB
    REDUCTION: 512/128 = 4× (exactly h/G = 32/8 = 4)

REDUCTION FACTOR = h/G
  G=1 (MQA): h× reduction
  G=h (MHA): 1× (no reduction)
  G=h/4:     4× reduction  ← sweet spot for quality/memory trade-off
```

## 4. Conversion from MHA to GQA

```
HOW TO CONVERT a pretrained MHA checkpoint to GQA?

Given MHA weights: W_K^1, ..., W_K^h  (h separate K projections)
Want GQA weights:  W_K^1_gqa, ..., W_K^G_gqa  (G groups)

METHOD — Mean Pooling (Ainslie et al. 2023):
  Group heads into G groups of h/G:
    Group g: {W_K^{(g-1)h/G+1}, ..., W_K^{g·h/G}}
  
  Average the K,V projections within each group:
    W_K^g_gqa = (G/h) × ∑_{i in group g} W_K^i   (mean of h/G matrices)
  
  MATHEMATICAL JUSTIFICATION:
    The averaged projection minimises the expected squared error:
    min_{W} E[‖X·W − X·W_K^i‖²]  averaged over i in group g
    → solution: W = mean(W_K^i for i in group g)

THEN FINE-TUNE:
  Brief SFT on ~5% of pre-training tokens
  → recovers most of the quality lost from averaging

QUALITY AFTER CONVERSION (Ainslie et al. 2023):
  Converted GQA ≈ trained-from-scratch GQA within 1-2% on benchmarks.
  This avoids the need to retrain large models!
```

## 5. Memory and Throughput Numbers

```
CONCRETE COMPARISON (LLaMA-3 8B):

                   MHA (h=32)   GQA (G=8)   MQA (G=1)
  W_K params:       16.8M         4.2M        0.52M
  W_V params:       16.8M         4.2M        0.52M
  KV per token:     512 KB       128 KB        16 KB
  4K context cache:  2.1 GB      537 MB        67 MB
  Max batch (80GB):  ~37          ~147          ~1000+
  Throughput gain:   1×           ~4×           ~30×

  ┌─────────────────────────────────────────────────────────────┐
  │  QUALITY vs MEMORY TRADE-OFF:                               │
  │                                                             │
  │  Quality                                                    │
  │  ↑ MHA  ●                                                  │
  │         │ ↖ diminishing returns                            │
  │     GQA (G=h/4) ●                                          │
  │              MQA ●   ← big quality drop                    │
  │                  ─────────────────────────────── Memory     │
  │                  Low                            High        │
  │                                                             │
  │  GQA hits the Pareto frontier:                              │
  │  Near-MHA quality with 4× memory reduction.                │
  └─────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. For h=40, G=5 (as in LLaMA-3 70B), compute the KV cache reduction factor relative to MHA and the KV cache size for 32K context (d_k=128, L=80 layers, BF16).
2. Derive the mean-pooling conversion formula from first principles. Show it minimises the expected squared distance between the converted and original K projections.
3. For a deployment serving Q=10⁶ queries/day with average 2K context: compute the total KV cache memory needed per day for MHA vs GQA (G=8) LLaMA-3 8B, assuming 100 concurrent requests.
