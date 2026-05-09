# Chapter 04 — Attention Mechanisms

The attention operation is the central innovation of the Transformer. This chapter derives it from
first principles, covers all modern variants, and explains the memory/compute trade-offs.

---

## Sub-topics

| # | Folder | Topic |
|---|--------|-------|
| 1 | `01_Scaled_Dot_Product/` | Core QKV formula, full derivation |
| 2 | `02_Multi_Head_Attention/` | Parallel heads, projection matrices |
| 3 | `03_Grouped_Query_Attention/` | GQA, MQA — reducing KV memory |
| 4 | `04_Causal_Masking/` | Autoregressive masking |
| 5 | `05_FlashAttention/` | IO-aware exact attention algorithm |

---

## Key Equations at a Glance

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

**Multi-Head Attention:**
```
MHA(X) = Concat(head₁, …, head_h) · W_O
  head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
```

**GQA** — G KV groups shared across h/G query heads each:
```
KV cache size ∝ G/h  of standard MHA
```

**Causal mask** (for position (i,j)):
```
M[i,j] = 0 if j ≤ i  else −∞
```

**FlashAttention** complexity:
```
Memory: O(N)   vs   O(N²) for standard attention
FLOPS:  O(N²d) — same, but with fewer HBM reads/writes
```
