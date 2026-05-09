# 03 — Learning Rate Schedules

## 1. Why the Learning Rate Schedule Matters

```
Too large LR:              Too small LR:          Good LR schedule:
  Loss                       Loss                   Loss
  │ /\/\/\/                  │\                     │\
  │/        (diverges)       │ ──────────────       │ ─\
  │                          │   (slow convergence) │   ─────────────
  └─────────── steps         └────────── steps      └──────── steps
```

## 2. Linear Warmup

```
η(t) = η_max × (t / T_warmup)   for t ≤ T_warmup

WHY WARMUP?
  At initialisation (t=0):
  - Weights are random (no useful representation yet)
  - Gradients are noisy and large
  - Large LR causes large random steps → divergence

  ┌─────────────────────────────────────────────────────────┐
  │  t=0:    η=0      (no update, stable)                    │
  │  t=100:  η=0.03η_max (gentle start)                     │
  │  t=1000: η=η_max  (full speed ahead)                    │
  │                                                         │
  │  LR                                                     │
  │  │         /── η_max                                    │
  │  │        /                                             │
  │  │       /                                              │
  │  │      /                                               │
  │  │─────/────────────────── steps                        │
  │  0   T_warmup                                           │
  └─────────────────────────────────────────────────────────┘

LLaMA-3: T_warmup = 2000 steps, η_max = 3e-4
```

## 3. Cosine Annealing

```
η(t) = η_min + 0.5(η_max − η_min)(1 + cos(π × (t − T_warmup)/(T − T_warmup)))

TRAJECTORY (T_warmup=1000, T=100000, η_max=3e-4, η_min=3e-5):

  LR (×10⁻⁴)
  3.0 ─ ─ ─ ─ ─ ─╮
                    ╮
  2.0 ─ ─ ─ ─ ─ ─  ╮
                      ╮
  1.0 ─ ─ ─ ─ ─ ─    ╮
                       ╮
  0.3 ─ ─ ─ ─ ─ ─ ─ ─ ╯─────────────────────
      │  warm │               cosine decay              │
  0   1K    10K              50K                      100K
             steps

KEY VALUES:
  t = T_warmup:          η = η_max  = 3.0e-4
  t = (T+T_warmup)/2:    η = (η_max+η_min)/2 = 1.65e-4
  t = T:                 η = η_min  = 0.3e-4

Cosine gives a smooth, graceful slowdown — better than abrupt step decay.
```

## 4. Warmup-Stable-Decay (WSD) Schedule

```
            ┌─────────────────────────────────────┐
  η_max  ───┤    Warmup   │   Stable   │  Decay   │
            │   /───────  │────────────│   ─╮      │
  η_min  ───┤  /          │            │    ╰─────  │
            └─────────────────────────────────────┘
               T_w        T_s                 T

  Phase 1 (Warmup): 0 to T_w        η linearly ramps to η_max
  Phase 2 (Stable): T_w to T_s      η = η_max  (constant)
  Phase 3 (Decay):  T_s to T        η cosine decays to η_min

WHY WSD?
  The stable phase can be EXTENDED without restarting the schedule.
  
  "I want to train 50% more tokens than planned":
  ─────────────────────────────────────────────────
  Cosine schedule:  must restart from scratch (schedule was hardcoded to T)
  WSD schedule:     just extend phase 2!
  
  Phase 2 at constant η_max:
    - Model learns steadily
    - Checkpoints are valid training intermediates
    - Can be resumed by any team at any time
    - Phase 3 (decay) is only run once at the final budget
```

## 5. LR Schedules Compared at Same Budget

```
  LR
  │╮                                  (warmup→cosine)
  ││╮
  │ ╰─────╮
  │        ╮
  │         ╰──────────────────────────
  │
  │╮                                  (WSD)
  ││╮
  │ ╰───────────────╮
  │                  ╲
  │                   ╰───────────────
  │
  └──────────────────────────────────── steps

FINAL PPL comparison (illustrative, LLaMA-3 scale):
  Cosine:       PPL = 6.42
  WSD:          PPL = 6.39  (slightly better, flexible)
  Step decay:   PPL = 6.51  (worse, abrupt transitions)
  Constant LR:  PPL = 7.10  (much worse, can't fine-tune at the end)
```

---

## Exercises

1. For T_warmup=2000, T=100000, η_max=3e-4, η_min=3e-5: Compute η at t=1000, t=50000, t=100000.
2. Why does cosine annealing reach η_min gradually rather than abruptly? What does this do for the loss trajectory?
3. In the WSD schedule, why is the stable phase important for continued training scenarios?
