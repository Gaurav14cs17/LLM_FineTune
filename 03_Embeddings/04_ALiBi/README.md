# 04 — ALiBi: Attention with Linear Biases

## 1. Core Idea

```
STANDARD ATTENTION (no position info in scores):
  score(q_t, k_s) = qₜᵀ kₛ / √d_k       ← same regardless of distance

ALIBI — add a distance penalty directly to the score:
  score(q_t, k_s) = qₜᵀ kₛ / √d_k  −  m × |t − s|
                    ──────────────    ───────────────
                    content score     position penalty

The farther away a key is, the more it's penalised.
```

## 2. The Bias Matrix Visualised

```
For head with slope m = 1/4, sequence length L = 6:

Positions:        s=0    s=1    s=2    s=3    s=4    s=5
                ┌──────┬──────┬──────┬──────┬──────┬──────┐
  q=0 (t=0)    │  0   │  -∞  │  -∞  │  -∞  │  -∞  │  -∞  │  (causal: future masked)
                ├──────┼──────┼──────┼──────┼──────┼──────┤
  q=1 (t=1)    │-0.25 │  0   │  -∞  │  -∞  │  -∞  │  -∞  │
                ├──────┼──────┼──────┼──────┼──────┼──────┤
  q=2 (t=2)    │-0.50 │-0.25 │  0   │  -∞  │  -∞  │  -∞  │
                ├──────┼──────┼──────┼──────┼──────┼──────┤
  q=3 (t=3)    │-0.75 │-0.50 │-0.25 │  0   │  -∞  │  -∞  │
                ├──────┼──────┼──────┼──────┼──────┼──────┤
  q=4 (t=4)    │-1.00 │-0.75 │-0.50 │-0.25 │  0   │  -∞  │
                ├──────┼──────┼──────┼──────┼──────┼──────┤
  q=5 (t=5)    │-1.25 │-1.00 │-0.75 │-0.50 │-0.25 │  0   │
                └──────┴──────┴──────┴──────┴──────┴──────┘

Diagonal = 0 (attending to self, no penalty)
Distance 1 = −0.25m penalty
Distance 5 = −1.25m penalty → very unlikely after softmax
```

## 3. Head Slopes — Different Attention Ranges

```
For h=8 heads, slopes form a geometric sequence:
  m = 2^{-8/h} = 2^{-1} = 0.5  as starting point

  Head 1:  m = 1/2   = 0.500   ← steep penalty, very local attention
  Head 2:  m = 1/4   = 0.250
  Head 3:  m = 1/8   = 0.125
  Head 4:  m = 1/16  = 0.0625
  Head 5:  m = 1/32  = 0.0313
  Head 6:  m = 1/64  = 0.0156
  Head 7:  m = 1/128 = 0.0078
  Head 8:  m = 1/256 = 0.0039  ← gentle penalty, long-range attention

EFFECT ON SOFTMAX:
  Head 1 (m=0.5):                  Head 8 (m=0.0039):
  After bias, attention looks:      After bias, attention looks:
  1.00 ██████████                  0.33 ████████████████████████████████
  0.82  ████████                   0.31 ██████████████████████████████
  0.67   ██████                    0.29 ████████████████████████████
  0.55    █████                    0.27 ██████████████████████████
  0.45     ████                    0.25 ████████████████████████
  ← focuses near        ← spreads far

Different heads learn different linguistic phenomena at different scales!
```

## 4. Why ALiBi Extrapolates to Longer Contexts

```
TRAINING at L=1024:
  Maximum distance seen: 1024
  Maximum penalty: m × 1024

TEST at L=4096:
  Distance 2000 appears for the first time
  Penalty: m × 2000  (just a larger value on the same linear scale)
  The model has SEEN the pattern "farther = less relevant" and extends it

COMPARE: Sinusoidal PE at position 1025:
  PE(1025) is a vector NEVER SEEN during training
  The model has no idea how to interpret it
  Performance degrades sharply beyond training length

ALiBi extrapolation on WikiText-103 (Press et al. 2022):
  Train length:  L = 1024
  Test  lengths: 1024   2048   4096
  ─────────────────────────────────
  Learned PE:    PPL=   +0.1   +4.2  (degrades quickly)
  ALiBi:         PPL=   +0.1   +0.3  (barely changes!)
```

---

## Exercises

1. For h=4, compute the 4 ALiBi slopes using the geometric formula.
2. For position t=10, key s=3, m=1/8, compute the ALiBi bias.
3. Why do different heads need different slopes rather than a single shared slope?
