# 02 — Language Model Definition

## 1. Formal Definition

```
A LANGUAGE MODEL is a probability distribution over token sequences:

  P_θ: V* → [0, 1]   (maps any finite sequence to a probability)

where V* = ∪_{n≥0} Vⁿ is the set of all finite sequences over vocabulary V.

VALID DISTRIBUTION requirements:
  (i)  P_θ(w₁,...,wₙ) ≥ 0            for all sequences
  (ii) ∑_{all sequences} P_θ(w) = 1  (normalises over all possible strings)

By the chain rule:
  P_θ(w₁,...,wₙ) = ∏_{t=1}^{n} P_θ(wₜ | w₁,...,wₜ₋₁)

So specifying a LM = specifying P_θ(next token | all previous tokens)

AUTOREGRESSIVE GENERATION:
  ┌────────────────────────────────────────────────────────────────┐
  │ Step 1: x = [BOS]                                              │
  │ Step 2: P_θ(w₁|BOS) → sample w₁                               │
  │ Step 3: P_θ(w₂|BOS,w₁) → sample w₂                            │
  │ ...                                                            │
  │ Step t: P_θ(wₜ|w₁,...,wₜ₋₁) → sample wₜ                      │
  │ Stop when wₜ = EOS                                             │
  └────────────────────────────────────────────────────────────────┘
```

## 2. Neural LM — Softmax Distribution

```
The Transformer produces logit vector z(context) ∈ ℝ^{|V|} for each context.

SOFTMAX converts logits to probabilities:
  P_θ(wₜ = k | context) = exp(zₖ) / ∑_{j=1}^{|V|} exp(zⱼ)

PROPERTIES of softmax:
  (i)  exp(zₖ) > 0  for all k → all probabilities positive ✓
  (ii) ∑_k exp(zₖ)/∑_j exp(zⱼ) = 1  → valid distribution ✓

NUMERICAL STABILITY — log-sum-exp trick:
  Direct computation: exp(z) can OVERFLOW for z > 88 (float32 max ≈ 3.4×10³⁸)
  
  Stable computation:
    m = max(z)   ← find the maximum logit
    log ∑_k exp(zₖ) = m + log ∑_k exp(zₖ − m)
                        ↑ now all exponents ≤ 0 → no overflow
  
  EXAMPLE:  z = [1000, 1001, 999]
    Direct: exp(1000) + exp(1001) + exp(999) = OVERFLOW ✗
    Stable: m=1001, compute exp(-1)+exp(0)+exp(-2) = 0.368+1.0+0.135 = 1.503
            ∑ = exp(1001) × 1.503  ← only one large number needed

GRADIENT OF SOFTMAX (critical for training):
  Let p = softmax(z), loss = −log p_{true} (cross-entropy)

  ∂loss/∂zₖ = pₖ − 1[k = true_token]
             = pₖ − yₖ   where y = one-hot target

  This CLEAN form is the reason cross-entropy + softmax is used universally.
  Gradient = prediction error.  Large error → large gradient → fast learning.
```

## 3. Negative Log-Likelihood Loss — Complete Derivation

```
MAXIMUM LIKELIHOOD ESTIMATION (MLE):
  Given corpus W = (w₁, w₂, ..., w_T), find θ* maximising:

  L_ML(θ) = log P_θ(w₁,...,w_T)
           = ∑_{t=1}^{T} log P_θ(wₜ | w₁,...,wₜ₋₁)   ← by chain rule

  θ* = argmax_θ L_ML(θ) = argmin_θ [−L_ML(θ)]

NEGATIVE LOG-LIKELIHOOD:
  NLL(θ) = −L_ML(θ) = −∑_{t=1}^{T} log P_θ(wₜ | w<t)

AVERAGED NLL (training loss):
  ┌──────────────────────────────────────────────────────────────┐
  │                   T                                          │
  │          1       ∑                                           │
  │ L(θ) = − ─  ×   ─  log P_θ(wₜ | w₁,...,wₜ₋₁)              │
  │          T      t=1                                          │
  └──────────────────────────────────────────────────────────────┘

WHY AVERAGE over T?
  Without averaging: loss scales with sequence length → different learning rates
  needed for different sequence lengths.
  With averaging: loss is per-token → consistent gradient magnitude regardless of T.

WHY LOG?
  (i) Converts product of probabilities to sum → numerically stable
      P(sequence) = ∏ P(wₜ|...) can be as small as 0.5^{1000} = 10^{-301}
      log P = ∑ log P(wₜ|...) stays in reasonable range

  (ii) Maximising ∏ P(wₜ) is equivalent to maximising ∑ log P(wₜ)
       (log is monotonically increasing, so argmax is preserved)
  
  (iii) log P is bounded above by 0:  log P ∈ (−∞, 0]
        so −log P ∈ [0, +∞) → suitable as a minimisation objective
```

## 4. Perplexity — Full Mathematical Treatment

```
DEFINITION:
  PPL(W) = exp( L(θ) )
         = exp( −(1/T) ∑_{t=1}^{T} log P_θ(wₜ | w<t) )

EQUIVALENT FORMS:
  PPL = exp(H_cross-entropy per token)
      = 2^{H_bits per token}    (in bits using log₂)
      = P_θ(W)^{−1/T}           (T-th root of inverse probability)

PROOF of P^{-1/T} form:
  PPL = exp(−(1/T) log P)
      = exp(log P^{-1/T})
      = P^{-1/T}   ■

INTUITION — "branching factor":
  If the model assigns uniform probability to B tokens at each step:
    P(wₜ) = 1/B  for each token
    L(θ) = −(1/T) ∑ log(1/B) = log B
    PPL = exp(log B) = B

  So PPL = B means "the model is as confused as if choosing uniformly from B options"

NUMERICAL EXAMPLE:
  4-token sequence: log-probs = [−0.5, −1.2, −0.3, −0.8]
  
  Step 1: Average = (0.5 + 1.2 + 0.3 + 0.8) / 4 = 0.7
  Step 2: PPL = exp(0.7) = 2.01

  "The model's effective branching factor is about 2 — it nearly always knows the next token."

PPL REFERENCE SCALE:
  PPL value │ Meaning
  ──────────┼──────────────────────────────────────────────
  PPL = 1   │ Perfect model (impossible without memorisation)
  PPL = 2   │ Excellent (~1 bit uncertainty per token)
  PPL = 10  │ Good (knows top 10 candidates well)
  PPL = 100 │ Struggling
  PPL = |V| │ Uniform — no better than random
  PPL = 50K │ For typical LLM vocab: completely random

  REAL MODEL PPLs (WikiText-103):
  GPT-2 (124M): 37  │  LLaMA-3 (8B): ~7  │  LLaMA-3 (70B): ~5
```

## 5. Cross-Entropy Loss vs NLL

```
For one token position t with true token y and model distribution q = P_θ(·|context):

  True distribution:  p = one-hot vector  (1 at index y, 0 elsewhere)
  Model distribution: q = softmax(z)

CROSS-ENTROPY:
  H(p, q) = −∑_k p(k) log q(k)
           = −log q(y)         ← since p(k) = 1 only at k=y
           = −log P_θ(y | context)
           = NLL for this token ✓

They are IDENTICAL when target is one-hot. Training cross-entropy = NLL.

RELATION TO KL DIVERGENCE:
  H(p, q) = H(p) + KL(p ‖ q)
  ↑                ↑       ↑
  cross-entropy   entropy  additional gap

  Since p is one-hot: H(p) = 0
  So H(p,q) = KL(p ‖ q) for one-hot targets.
  
  Minimising NLL ≡ minimising KL(p_true ‖ p_model) ← makes the model match the data distribution.
```

---

## Exercises

1. Show that for a uniform model PPL = |V|. Use the formula PPL = exp(−(1/T) ∑ log P) with P = 1/|V|.
2. Given a 4-token sequence with log-probs −1, −2, −1, −3, compute L and PPL step by step.
3. If you compress text with an LM of PPL P, the optimal compression rate is log₂(P) bits/token. For PPL=30, how many bits per token? Compare to ASCII encoding (8 bits/char ≈ 1.5 chars/token → ~12 bits/token).
