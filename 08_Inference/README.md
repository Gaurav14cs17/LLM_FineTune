# Chapter 08 — Inference and Decoding

Training teaches \(P_\theta(w_t\mid w_{<t})\); **inference** turns those probabilities into discrete token choices Greedy search, sampling with temperature, nucleus **top-p**, KV-cache reuse, and **speculative decoding** determine latency, diversity, and hardware cost at deployment.

```
  prompt tokens  ──►  prefill (parallel)  ──►  hidden states + KV cache seeded
                           │
 decode loop  ◄───────────┘  (one new token per step, unless spec-decoding)
      │
      ├── greedy / beam: argmax paths
      └── stochastic: sample ~ softmax(z / T) then post-filters (top-k, top-p, min-p)

  KV cache memory ∝ layers × heads_or_groups × d × sequence positions
```

## Sub-chapters

**`01_Greedy_and_Beam_Search/`**  
Greedy argmax is cheap but brittle; **beam search** retains top-\(B\) hypotheses by cumulative log-prob; contrasts with open-ended creative tasks where stochasticity helps.

**`02_Sampling/`** and **`02_Sampling_Strategies/`**  
Temperature sharpens/softens distributions; **top-k** and **top-p** truncate support; **min-p** adapts floors to logits scale—important for long-tail vocab-heavy domains.

**`03_KV_Cache/`**  
Stores attention keys/values for prior tokens so each new step is \(O(L)\) attention over history instead of quadratic **recompute**; analyses memory formula vs batch size and context length.

**`04_Speculative_Decoding/`**  
Runs a smaller **draft** model for \(k\) steps, then a **target** model batched-verifies; acceptance uses probability ratios; yields wall-clock speedups when draft aligns with target.

## Key formulas and concepts

**Greedy**

\[
w_t = \arg\max_w\ P_\theta(w\mid w_{<t}).
\]

**Temperature**

\[
P_T(w) \propto \exp(z_w / T),\quad T<1 \Rightarrow \text{sharper}.
\]

**Top-p (nucleus)**

Choose smallest set \(V\) s.t. \(\sum_{w\in V} P(w)\ge p\); renormalise on \(V\).

**KV cache bytes (order of magnitude)**

\[
\text{Memory} \sim L_{\mathrm{layers}} \times \text{tokens} \times (\text{KV width}) \times \text{dtype bytes} \times (\text{parallel batches}).
\]

**Speculative acceptance (per token)**

\[
P(\text{accept } w)=\min\Bigl(1,\ \frac{P_{\mathrm{target}}(w)}{P_{\mathrm{draft}}(w)}\Bigr)\ \text{(standard coupling story)}.
\]

## Prerequisites

- Chapter 04 (what KV tensors are).
- Softmax and categorical sampling intuition.

## Duplicate sampling folders

`02_Sampling` vs `02_Sampling_Strategies` overlap—pick one per curriculum pass; content goals align.

## Study tips

- Plot token entropy across decode steps: prompts often low-entropy; answers may need higher temperature.
- Benchmark **prefill vs decode** separately—optimisation targets differ (tensor-core GEMMs vs memory-bound small steps).

## Latency checklist

| Stage | Bottleneck often | Mitigation |
|-------|------------------|------------|
| Prefill | big matmuls | FlashAttention, long ctx kernels |
| Decode | KV bandwidth | GQA, quantised KV, paged caches |
| End-to-end | sampling overhead | fused softmax, efficient RNG |

## Quality vs diversity

- Lower temperature improves **factuality** on some QA tasks but increases **repetition** in open-ended generation—schedule temperature **per task** when building APIs.

## Structured outputs

- When decoding JSON or tool calls, combine **grammar masking** or constrained decoding (out of scope here) with temperature/top-p—the probability simplex tweaks alone may be insufficient.

## Intended outcomes

You should be able to: (1) derive KV RAM from a concrete config (layers, heads, width, dtype); (2) choose greedy/beam/sampling for a task profile; (3) explain speculative decoding **acceptance** intuition; (4) separate **prefill** vs **decode** bottlenecks when reading a GPU trace.

## Exam-style checks

- Write KV cache memory \(\propto\) batch \(\times\) layers \(\times\) sequence \(\times\) KV dim; identify where **GQA** enters the KV dim.
- When does raising temperature **reduce** repetitive loops—when does it **increase** factual errors on closed QA?

---

_End of chapter overview._
