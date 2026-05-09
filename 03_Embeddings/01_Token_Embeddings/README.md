# 01 — Token Embeddings

## 1. Formal Definition

A vocabulary V = {0, 1, …, |V|−1} maps each token to an integer ID.

The **embedding matrix** E ∈ ℝ^{|V| × d} assigns each token a real-valued vector:

```
E = ┌ e₀ ─────────────────── ┐   ← row 0: embedding of token 0 ("the")
    │ e₁ ─────────────────── │   ← row 1: embedding of token 1 ("a")
    │ e₂ ─────────────────── │   ← row 2: embedding of token 2 ("cat")
    │ ⋮                       │
    └ e_{|V|-1} ────────────  ┘   ← row |V|-1: embedding of last token

Each row eₜ ∈ ℝ^d is a d-dimensional real vector.

LOOKUP (forward pass):
  Input:  token_id  t  (a scalar integer)
  Output: x_t = E[t, :]   ∈ ℝ^d

  This is equivalent to a one-hot multiply:
    one_hot_t ∈ {0,1}^{|V|}  (1 at position t, 0 elsewhere)
    x_t = one_hot_t · E  =  E[t, :]
  but implemented as a direct index (O(1) lookup, not O(|V|·d) multiply).
```

## 2. Geometric Interpretation

```
Embedding space ℝ^d (shown in 3D for illustration):

         "king"  •
                   \
        "queen" •   \  direction of "royalty"
                     \
         "man"  •    •  "woman"
                    ↗
                  direction of "gender"

Key properties learned from data:
  ‖E["king"] − E["man"] − E["woman"] + E["queen"]‖ ≈ 0
  (famous word2vec analogy: king − man + woman ≈ queen)

  Cosine similarity:
  cos(eᵢ, eⱼ) = (eᵢ · eⱼ) / (‖eᵢ‖ · ‖eⱼ‖) ∈ [−1, 1]
  
  cos("cat", "kitten") ≈ 0.85  (similar meaning)
  cos("cat", "democracy") ≈ 0.02 (unrelated)

  Visual (2D projection of d=4096 embedding space):
  ┌────────────────────────────────────────────────────┐
  │                    · car                            │
  │         · truck               · bus                │
  │    · vehicle                                        │
  │                          · plane  · jet            │
  │  ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄     │
  │              · cat · dog · animal                  │
  │         · "the" · "a" · "an"   (function words)    │
  └────────────────────────────────────────────────────┘
  (Clusters form naturally: vehicles, animals, determiners)
```

## 3. Weight Tying — Full Mathematical Derivation

```
STANDARD (no tying): two separate matrices
  E_in  ∈ ℝ^{|V| × d}     (input embedding)
  E_out ∈ ℝ^{d × |V|}     (output projection / unembedding)
  
  Forward pass:
    x_t = E_in[t, :]                            ← embed
    h   = TransformerLayers(x_1, …, x_t)        ← process
    logits = h · E_out  ∈ ℝ^{|V|}              ← predict next token
    P(next = k) = softmax(logits)_k
  
  Parameters: 2 × |V| × d

WEIGHT TYING: E_out = E_inᵀ
  logits_k = h · E_in[k, :] = hᵀ · e_k
  
  Geometric meaning:
  logit_k = inner product between the hidden state h and token k's embedding e_k
  
  The model scores "how similar is my current hidden state to each token's embedding?"
  This is conceptually correct: the model produces a hidden state "pointing toward" the
  likely next token in embedding space.

  PROOF that tying doesn't hurt expressiveness:
    If we set E_out = E_in^T, the model can still:
    - Learn arbitrary output distributions via the hidden state h
    - The final LayerNorm before lm_head can re-scale h independently
    - In practice, quality is equal or better (Press & Wolf 2017)
  
  PARAMETERS SAVED:
  Without tying: |V| × d  +  |V| × d  = 2|V|d  = 2 × 128K × 4096 = 1.07B
  With tying:    |V| × d               =  |V|d  = 524M
  SAVED:         524M params  (6.5% of the 8B total)
```

## 4. Embedding Scale — Why √d?

```
PROBLEM: After RoPE and positional encoding, we ADD two vectors:
  z_t = x_t + PE(t)     (original Transformer)

For this addition to be meaningful, both terms should have similar magnitude.

SINUSOIDAL PE has entries bounded in [−1, 1]:
  ‖PE(t)‖ = √(d/2)   (each sin/cos entry has magnitude ≤ 1, d/2 pairs)
            = √(d/2) ≈ √(d) in order of magnitude

EMBEDDING (standard init with std=1/√d):
  E[t] ~ N(0, 1/d)   → ‖E[t]‖ ≈ √( d × 1/d ) = 1

WITHOUT scaling:  ‖x_t‖ ≈ 1  vs  ‖PE(t)‖ ≈ √d
  → Positional encoding DOMINATES at large d → token identity lost!

WITH scaling by √d:  x_t = E[t] × √d → ‖x_t‖ ≈ √d
  → Both terms have magnitude ≈ √d → balanced addition ✓

Example for d = 512:
  ‖E[t]‖ ≈ 1   without scaling
  ‖PE(t)‖ ≈ 16  (very large relative to embedding!)
  After scaling: ‖x_t‖ ≈ 16  ← matched
```

## 5. Initialisation — Mathematical Justification

```
GOAL: At initialisation, the embedding layer output should have unit variance.

For E[t] ~ N(0, σ²) (each element independently):
  x_t,i = E[t, i]  has variance σ²

  ‖x_t‖² = ∑_{i=1}^{d} x_t,i²  has expectation d × σ²
  
  ‖x_t‖  ≈  σ√d   (by concentration of measure / law of large numbers)

TYPICAL CHOICES:
  σ = 1/√d  → ‖x_t‖ ≈ 1      (unit-norm embeddings at init)
  σ = 0.02  → ‖x_t‖ ≈ 0.02√d  (LLaMA-3: fixed small std)

Why LLaMA uses σ = 0.02:
  ‖x_t‖ ≈ 0.02 × √4096 = 0.02 × 64 = 1.28
  Combined with RMSNorm (normalises to unit RMS), the exact init matters less.

GRADIENT FLOW at INIT (important for learning rare tokens):
  Token t appears N_t times in training data.
  Gradient per step: ∂L/∂E[t] = (1/N_t) × (attention-weighted feedback)
  
  For rare tokens (N_t = 100 in 1T corpus):
  Effective updates per epoch:  ~100   (very sparse learning signal)
  
  Large embedding tables need careful LR tuning or separate LR for embedding vs rest.
```

## 6. Parameter Count and Memory

```
EMBEDDING TABLE:
  |V| × d × bytes_per_element

  LLaMA-3 8B:
    |V| = 128,000   d = 4,096   BF16 = 2 bytes
    = 128,000 × 4,096 × 2 = 1,048,576,000 bytes ≈ 1.05 GB

  LLaMA-3 8B total params ≈ 8B × 2 bytes = 16 GB
  Embedding fraction: 1.05 / 16 = 6.6%

COMPARISON ACROSS MODELS:
  Model          |V|      d      Emb params   % of total
  ──────────────────────────────────────────────────────
  GPT-2          50,257  1,024    51M          8.5%
  LLaMA-2 7B     32,000  4,096   131M          1.9%
  LLaMA-3 8B    128,000  4,096   524M          6.5%
  GPT-4 (est.)  100,000  ~12,288 1.2B          ~1%

  LLaMA-3 doubled the vocab size (32K → 128K) for better multilingual coverage.
  Cost: 393M extra embedding params (manageable vs 8B total).
```

---

## Exercises

1. For |V|=32000 and d=4096, compute the size of the embedding table in MB (float32 = 4 bytes).
2. Derive why weight tying (E_out = E_in^T) is geometrically sensible by interpreting the logit computation as cosine similarity (up to normalisation).
3. If you set d=64 for |V|=50000, compute how many tokens would need to be approximately orthogonal. Since ℝ^64 only has 64 truly orthogonal directions, what does this imply about the representation capacity?
