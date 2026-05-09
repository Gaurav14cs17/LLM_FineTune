# Chapter 06 — Pre-training

Pre-training is large-scale minimisation of the **causal language modelling** objective over trillions of tokens. This chapter spells out the **loss**, **AdamW** dynamics, **learning-rate schedules**, and **mixed precision** numerics that make billion-parameter training stable on modern accelerators.

```
  data stream  ──►  micro-batches  ──►  forward+backward
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │  AdamW states   │  BF16/FP16 matmul │  cosine LR   │
                    │  (moments)      │  loss scaling      │  warmup      │
                    └────────────────────────────────────────────────────┘
                                         │
                     gradients  ──►  weight update  ──►  next step
```

## Sub-chapters

**`01_Objective_and_Loss/`**  
Canonical causal LM NLL, label shifting, packing multiple documents with attention masks / special tokens, and bookkeeping when using **document boundaries** vs pure token streams.

**`01_AdamW/`, `02_AdamW/`, `02_AdamW_Optimizer/`** (overlapping duplicates in repo)  
Adam moments \(\beta_1,\beta_2\), bias correction, **decoupled weight decay** vs L2, and practical hyper-parameters (\(\epsilon\), gradient clipping) seen in LLM recipes.

**`03_LR_Schedules/`**  
Warmup stabilises early optimisation; cosine decay and **WSD** (warmup-stable-decay) trade interpretability for throughput; relates max LR to batch scale (linear scaling rule sketches).

**`04_Mixed_Precision/`**  
BF16/FP16 forward, master weights in FP32 optionally, **loss scaling** when FP16 dynamic range is tight, and **gradient scaling** interactions with AdamW.

## Key formulas and concepts

**Training loss (single stream)**

\[
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{<t}).
\]

**AdamW (schematic)**

\[
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t,\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta \lambda \theta_t.
\end{aligned}
\]

**Cosine schedule (one form)**

\[
\eta(t)=\eta_{\min}+\tfrac{1}{2}(\eta_{\max}-\eta_{\min})\bigl(1+\cos(\pi t/T)\bigr).
\]

**Compute anchor (Chinchilla-style back-of-envelope)**

\[
C \approx 6 N D \quad \text{(dense TFLOPs proxy; } N \text{ params, } D \text{ tokens).}
\]

## Prerequisites

- Chapter 01 (objective = cross-entropy).
- Chapter 05 (what parameters \(\theta\) enumerate).
- Optional: basic SGD intuition.

## Folders with duplicate names

Multiple AdamW directories exist for historical layout; follow the newest path your course maintainer recommends—math is identical.

## Study tips

- Watch **gradient norm** traces alongside LR; divergent loss often precedes visible NaNs in BF16 runs.
- Reconcile \(C \approx 6ND\) with your framework's **reported** TFLOPs (attention, MoE, and pipeline parallelism change constants).

## Batch scaling intuition

- Linear LR scaling (increase LR with global batch) is heuristic—always validate with a short **warm LR sweep**; outliers exist at multi-thousand GPU scale.
- Gradient accumulation **simulates** larger batch without more memory for weights, but activation memory still grows with micro-batch unless checkpointing.

## Checkpoint hygiene

- Save **optimizer states** if resuming long runs; BF16 weights alone lose precision to continue Adam faithfully unless FP32 master copies kept separately.

## Data pipeline reminders

- **Deterministic dataloaders** matter for ablations comparing LR schedules; nondeterminism hides small-but-real perplexity deltas at scale.
- **Shuffle buffer** sizing interacts with curriculum; broken shuffles create loss spikes mistaken for instabilities.

## Intended outcomes

You should be able to: (1) write down the per-token objective and masking rules for packed sequences; (2) state AdamW **moment** updates and **bias correction**; (3) compare cosine vs WSD schedules at a high level; (4) explain why FP16 may still need **loss scaling** in legacy stacks while BF16 often does not.

## Exam-style checks

- Estimate total training FLOPs from published \(N\) and \(D\); identify which **constant** (e.g. 6 vs 8) your stack’s profiler uses and why it differs.
- List three signals that optimisation diverged **before** NaNs appear in BF16 training.

---

_End of chapter overview._
