# Chapter 08 — Inference & Decoding

Inference converts the model's probability distribution into text. The decoding strategy
dramatically affects output quality, diversity, and speed.

---

## Sub-topics

| # | Folder | Topic |
|---|--------|-------|
| 1 | `01_Greedy_and_Beam_Search/` | Deterministic decoding |
| 2 | `02_Sampling_Strategies/` | Temperature, top-k, top-p, min-p |
| 3 | `03_KV_Cache/` | Caching key/value pairs for fast generation |
| 4 | `04_Speculative_Decoding/` | Draft + verify for 2-4× speedup |

---

## Key Equations at a Glance

**Greedy:**
```
wₜ = argmax_k P_θ(wₜ = k | w<t)
```

**Beam search:**
```
Keep top-B hypotheses at each step by cumulative log-prob
```

**Temperature scaling:**
```
P_T(w) = softmax(z / T)     T<1: sharper   T>1: flatter
```

**Top-p (nucleus) sampling:**
```
V_p = smallest set V such that ∑_{w∈V} P(w) ≥ p
Sample from V_p after renormalisation
```

**KV cache memory:**
```
Size = 2 × L_layers × d × context_length × bytes
```

**Speculative decoding acceptance:**
```
P(accept token t) = min(1, P_target(t) / P_draft(t))
```
