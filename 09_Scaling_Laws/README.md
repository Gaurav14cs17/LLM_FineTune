# Chapter 09 — Scaling Laws & Emergent Capabilities

Scaling laws describe how LLM loss, capability, and compute interact as model size,
data, and compute increase. They guide every major training decision.

---

## Sub-topics

| # | Folder | Topic |
|---|--------|-------|
| 1 | `01_Kaplan_Power_Laws/` | Kaplan et al. (2020) — original power laws |
| 2 | `02_Chinchilla/` | Hoffmann et al. (2022) — compute-optimal training |
| 3 | `03_Emergent_Capabilities/` | Phase transitions, emergence, benchmarks |

---

## Key Equations at a Glance

**Kaplan loss power law:**
```
L(N) ≈ (N_c/N)^{α_N}     α_N ≈ 0.076
L(D) ≈ (D_c/D)^{α_D}     α_D ≈ 0.095
L(C) ≈ (C_c/C)^{α_C}     α_C ≈ 0.050
```

**Chinchilla compute-optimal allocation:**
```
N_opt ∝ C^{0.5},   D_opt ∝ C^{0.5}
N_opt = 0.1206 · C^{0.4845}
D_opt = 5.4 · N_opt  (approximately D = 20N tokens)
```

**FLOPs estimation:**
```
C ≈ 6ND    (forward + backward, ignoring attention)
```

**Perplexity gain from 2× more data (Chinchilla):**
```
ΔL ≈ −α_D · log(2) / log(D_c/D)
```
