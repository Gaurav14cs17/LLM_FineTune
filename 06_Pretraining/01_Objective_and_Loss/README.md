# 01 — Pre-training Objective and Loss

## 1. The Causal LM Objective — Formal Derivation

```
DATASET: corpus W = (w₁, w₂, ..., w_T)  (concatenated documents)

MAXIMUM LIKELIHOOD OBJECTIVE:
  max_θ  log P_θ(w₁, ..., w_T)

  = max_θ  log ∏_{t=1}^{T} P_θ(wₜ | w₁, ..., wₜ₋₁)   ← chain rule

  = max_θ  ∑_{t=1}^{T} log P_θ(wₜ | w₁, ..., wₜ₋₁)   ← log(∏) = ∑(log)

AVERAGE NLL TRAINING LOSS (normalised by length):
  ┌────────────────────────────────────────────────────────────────────┐
  │             T                                                      │
  │      1     ∑                                                       │
  │ L = − ─  × ─  log P_θ(wₜ | w₁, ..., wₜ₋₁)                       │
  │      T    t=1                                                      │
  └────────────────────────────────────────────────────────────────────┘

LOSS RANGE:
  Perfect model:  P(correct token) = 1  →  log P = 0  →  L = 0
  Uniform model:  P(correct token) = 1/|V|  →  log P = −log|V|  →  L = log|V|
  For |V|=128K:   L_max = log(128000) ≈ 11.76 nats ≈ 17 bits

CONVERGENCE:
  LLaMA-3 8B achieves L ≈ 1.5 nats/token (PPL ≈ 4.5) on clean text
  L_min (language entropy) ≈ 0.7 nats/token
  Gap remaining ≈ 0.8 nats (room for better models)
```

## 2. Softmax Cross-Entropy — Step-by-Step Derivation

```
For token position t with context, the model produces logits z ∈ ℝ^{|V|}.
Let y = w_t (true token, an integer index).

FORWARD:
  Step 1: Softmax probabilities
    p_k = exp(z_k) / ∑_{j} exp(z_j)   for k = 0, 1, ..., |V|−1

  Step 2: NLL loss for this token
    loss_t = −log p_y
           = −z_y + log ∑_j exp(z_j)
           = log_sum_exp(z) − z_y

  Where: log_sum_exp(z) = log ∑_j exp(z_j)  (the log-normaliser)

NUMERICAL EXAMPLE:
  |V| = 4,  z = [2.0, 1.5, 0.3, -0.5],  y = 0 (true token is index 0)
  
  exp(z): [7.39, 4.48, 1.35, 0.61]
  sum:    13.83
  p: [0.534, 0.324, 0.098, 0.044]   (sum = 1.0 ✓)
  loss = −log(0.534) = 0.627 nats
  
  ALTERNATIVE (numerically stable):
  m = max(z) = 2.0
  shifted: [0, -0.5, -1.7, -2.5]
  exp:     [1.0, 0.607, 0.183, 0.082]
  sum = 1.872
  log_sum_exp = 2.0 + log(1.872) = 2.0 + 0.627 = 2.627
  loss = 2.627 − 2.0 = 0.627 nats  (same result, no overflow!)
```

## 3. Gradient of Cross-Entropy Loss

```
∂loss_t/∂z_k = ?

Using loss_t = log_sum_exp(z) − z_y:

  ∂ log_sum_exp(z) / ∂z_k = exp(z_k) / ∑_j exp(z_j) = p_k

  ∂(−z_y) / ∂z_k = −1[k = y]   (indicator: 1 if k=y, else 0)

COMBINING:
  ┌──────────────────────────────────────────────────────────────┐
  │  ∂loss_t/∂z_k = p_k − 1[k = y] = p_k − y_k               │
  │                                                              │
  │  where y = one-hot vector (y_k = 1 if k=true, else 0)       │
  └──────────────────────────────────────────────────────────────┘

INTERPRETATION: gradient = prediction error!
  For k = true token:   gradient = p_y − 1 < 0  (push logit UP)
  For k ≠ true token:   gradient = p_k > 0       (push logit DOWN)

EXAMPLE (from above, y=0):
  ∂loss/∂z₀ = 0.534 − 1 = −0.466  (increase z₀ to raise P(0))
  ∂loss/∂z₁ = 0.324 − 0 = +0.324  (decrease z₁ to lower P(1))
  ∂loss/∂z₂ = 0.098 − 0 = +0.098  (decrease z₂)
  ∂loss/∂z₃ = 0.044 − 0 = +0.044  (decrease z₃)
  
  Sum of gradients = (0.534−1) + 0.324 + 0.098 + 0.044 = 0 ✓ (must sum to 0)
```

## 4. Loss Decomposition — What the Loss Measures

```
TOTAL LOSS can be decomposed into contributions:
  L = (1/T) ∑_t loss_t

  High loss tokens: words the model finds surprising
  Low loss tokens:  words the model predicts confidently

EXAMPLE SENTENCE: "The Eiffel Tower is located in Paris France"
  Token    │ Loss (nats)  │ Interpretation
  ─────────┼──────────────┼────────────────────────────────────────
  "The"    │ 0.2          │ BOS → "The" is very common
  "Eiffel" │ 2.8          │ "The" → "Eiffel" is somewhat rare
  "Tower"  │ 0.3          │ "Eiffel" → "Tower" is highly predictable
  "is"     │ 0.4          │ "Tower" → "is" is common grammar
  "located"│ 1.2          │ somewhat predictable with context
  "in"     │ 0.5          │ "located" → "in" very predictable
  "Paris"  │ 0.9          │ correct but many cities are possible
  "France" │ 0.6          │ "Paris" → "France" fairly predictable
  ─────────────────────────────────────────────────────────────────
  Average  │ 0.86 nats   │ PPL ≈ exp(0.86) ≈ 2.4

INFORMATION THEORY VIEW:
  loss_t = −log₂ P_θ(wₜ) bits = optimal compression length for token t
  Average loss = average bits per token = description length per token
  Minimising NLL = maximising compression rate
```

## 5. Effect of Document Boundaries

```
PROBLEM: documents are concatenated for efficiency:
  [doc1_tokens] [doc2_tokens] [doc3_tokens] ... (packed into length-L chunks)

WITHOUT boundary handling:
  The model at the start of doc2 sees tokens from doc1 as context.
  P(doc2_start | doc1_tokens) is evaluated — nonsensical cross-document prediction!

WITH document separation tokens:
  [doc1] <|eot_id|> [doc2] <|eot_id|> [doc3]  ← LLaMA-3 approach

  The model learns that after <|eot_id|>, context resets.
  Cross-document attention is masked to 0:
  
  ┌─────────────────────────────────────────────────────────────┐
  │  doc1 tokens    │ eot │  doc2 tokens   │ eot │  doc3 tokens │
  │  ← attend freely →   │ ← attend freely →    │ ← attend     │
  │  (cross-doc: masked)  │ (cross-doc: masked)  │  freely →    │
  └─────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. Compute the gradient ∂loss/∂z for z=[1,2,3] and target=index 2 (the third token). Show all steps.
2. Show that minimising cross-entropy loss is equivalent to maximising likelihood by connecting the two objectives mathematically.
3. Why is `log_sum_exp` used instead of `log(sum(exp(z)))` for numerical stability? Give a concrete example where the naive computation overflows in float32 (max ≈ 3.4×10³⁸).
