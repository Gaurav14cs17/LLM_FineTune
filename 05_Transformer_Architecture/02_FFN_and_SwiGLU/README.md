# 02 — Feed-Forward Network and SwiGLU

## 1. Standard FFN — Role and Formulation

```
The FFN processes each token position INDEPENDENTLY (no cross-token interaction):
  FFN(X) applies the same function to each row of X ∈ ℝ^{n×d}

For a single token x ∈ ℝ^d:
  STANDARD (GELU/ReLU):
    FFN(x) = σ(x·W₁ + b₁) · W₂ + b₂
    W₁ ∈ ℝ^{d×d_ff},   b₁ ∈ ℝ^{d_ff}
    W₂ ∈ ℝ^{d_ff×d},   b₂ ∈ ℝ^d

WHY 4× EXPANSION (d_ff = 4d)?
  UNIVERSAL APPROXIMATION: a 2-layer network with n hidden units can approximate
  any continuous function on a compact domain.
  
  The FFN can be viewed as a KEY-VALUE MEMORY (Geva et al. 2021):
    Each hidden neuron i corresponds to a (pattern, value) pair:
      pattern_i = W₁[:,i]   (first column of W₁ = a pattern that activates neuron i)
      value_i   = W₂[i,:]   (corresponding row of W₂ = what to output when activated)
    
    FFN(x) = ∑_i σ(x·pattern_i) × value_i
           = weighted sum of stored values (soft lookup table!)
  
  Larger d_ff → more stored patterns → more factual knowledge capacity.
  Research shows FFN neurons store factual knowledge (e.g., "Paris is the capital of France").
```

## 2. SwiGLU — Derivation and Properties

```
SwiGLU (Shazeer 2020) = Swish activation + Gated Linear Unit

SWISH ACTIVATION:
  Swish(x) = x · σ(x)  where σ(x) = 1/(1+e^{-x})

  PROPERTIES:
    Swish(0) = 0 × 0.5 = 0
    Swish(x) → x as x → ∞   (linear for large positive x)
    Swish(x) → 0 as x → −∞  (gates off large negative inputs)
    
    Derivative:
    d/dx [x·σ(x)] = σ(x) + x·σ(x)(1−σ(x)) = σ(x)(1 + x − x·σ(x))
    → Smooth, non-monotone near x=−1.28 (small negative dip, helps with gradient flow)

GATED LINEAR UNIT (GLU):
  GLU(x) = (x·W₁) ⊙ σ(x·W₂)
  ↑ linear transformation  ↑ sigmoid gate

SWIGLU combines them:
  ┌──────────────────────────────────────────────────────────────┐
  │  SwiGLU(x) = (x · W₁ ⊙ Swish(x · W_gate)) · W₂            │
  │                                                              │
  │  = (x·W₁ ⊙ (x·W_gate · σ(x·W_gate))) · W₂                 │
  └──────────────────────────────────────────────────────────────┘

  W₁, W_gate ∈ ℝ^{d×d_ff}   (two "up" projections)
  W₂ ∈ ℝ^{d_ff×d}           (one "down" projection)
  THREE matrices total  (vs TWO for standard FFN)
```

## 3. Parameter Budget — Why d_ff = 2/3 × 4d for SwiGLU

```
STANDARD FFN (2 matrices):
  d × d_ff + d_ff × d = 2 × d × d_ff  parameters
  For d_ff = 4d:  2 × d × 4d = 8d²  parameters

SWIGLU FFN (3 matrices):
  d × d_ff  (W₁: gate)
  d × d_ff  (W_gate: gate signal)
  d_ff × d  (W₂: down)
  = 3 × d × d_ff  parameters (for square down projection)
  
  TO MATCH standard FFN parameter count:
    3 × d × d_ff = 8d²
    d_ff = 8d²/(3d) = (8/3)d ≈ 2.67d
  
  LLaMA uses d_ff = 8d/3 rounded to the nearest multiple of 256:
    d = 4096 → 8×4096/3 = 10922.7 → rounded → 10880 or 11008
    
  LLaMA-3 8B uses d_ff = 14336:
    → 14336/4096 = 3.5×  (bigger than 2.67× for capacity reason)
    Parameter count check: 3 × 4096 × 14336 = 176.2M per layer
                          × 32 layers = 5.64B (in FFN alone → 70% of model!)

ACTIVATION PATTERN COMPARISON:
  Position:     -4   -3   -2   -1    0    1    2    3    4
  ReLU:          0    0    0    0    0   1.0  2.0  3.0  4.0
  GELU:        ~0   ~0  -0.05  -0.16  0  0.84  1.95  3.0  4.0
  Swish:       ~0   ~0  -0.09  -0.28  0  0.73  1.76  2.86  3.93
  
  Swish has a SMALL NEGATIVE REGION near x ≈ −1.28 (minimum ≈ −0.28)
  This allows neurons to "vote down" features slightly, not just gate them off.
  Empirically: SwiGLU > GELU > ReLU for large language models.
```

## 4. Information Flow Through FFN

```
COMPOSITION WITH ATTENTION IN TRANSFORMER LAYER:
  
  Input:  x ∈ ℝ^d  (token representation after attention)

  GATE PATH:   x → x·W_gate → Swish(·) → swish_scores ∈ ℝ^{d_ff}
  LINEAR PATH: x → x·W₁    →              gate_values ∈ ℝ^{d_ff}
  HADAMARD:    hidden = swish_scores ⊙ gate_values ∈ ℝ^{d_ff}
  DOWN PROJ:   output = hidden · W₂ ∈ ℝ^d

VISUAL (d=4, d_ff=8 for illustration):
  Input x=[x₁,x₂,x₃,x₄]
        │          │
        │ W₁        │ W_gate
        ↓          ↓
  [h₁,...,h₈]  [g₁,...,g₈]  (expanded to d_ff=8)
        │          │ Swish(·)
        │          ↓
  h⊙Swish(g) = [h₁·σ̃(g₁),...,h₈·σ̃(g₈)]   (selective information flow!)
               │
               │ W₂  (project back to d=4)
               ↓
  output ∈ ℝ^4

INTERPRETATION:
  Each of the d_ff neurons represents a "fact" or "feature pattern".
  The gate determines WHICH patterns are relevant for this token.
  The linear path determines WHAT value those patterns contribute.
  → Dynamic routing: different tokens activate different neuron subsets.
```

## 5. Key-Value Memory Interpretation — Mathematical Proof

```
CLAIM (Geva et al. 2021): FFN(x) = ∑_i α_i · v_i  (weighted sum of "memory values")

PROOF for SwiGLU:
  output = (x·W₁ ⊙ Swish(x·W_gate)) · W₂
         = ∑_{i=1}^{d_ff} [(x·W₁)_i × Swish((x·W_gate)_i)] × W₂[i,:]
  
  Let:
    keys_1 = W₁[:,i]  (column i of W₁, shape ℝ^d)
    keys_g = W_gate[:,i]  (column i of W_gate)
    value_i = W₂[i,:]  (row i of W₂, shape ℝ^d)
  
  Then:
    (x·W₁)_i = x · keys_1_i     (match query x against key_1)
    (x·W_gate)_i = x · keys_g_i (match query x against gate key)
    α_i = (x·keys_1_i) × Swish(x·keys_g_i)   (gated coefficient)
  
  output = ∑_i α_i × value_i   ■

NEURON ANALYSIS:
  Neurons in middle layers ≈ factual recall
  ("Paris" concept → activates "capital city" neuron → outputs "France" direction)
  
  Neurons in last layers ≈ task-specific processing
  ("The answer is" → activates "enumerate" neuron → structures output format)
  
  ~70% of model parameters live in FFN layers (for d_ff=3.5×d)
  FFN = the model's "knowledge store"
  Attention = the model's "reasoning / routing" mechanism
```

---

## Exercises

1. Derive the parameter count ratio (SwiGLU vs standard ReLU FFN) and show that setting d_ff = 8d/3 makes them equal.
2. Prove that Swish(x) = x·σ(x) is smooth (infinitely differentiable) and compute d/dx[Swish(x)].
3. In the key-value memory interpretation, how many "memories" are stored in LLaMA-3 8B's FFN? (32 layers × d_ff = 14336 neurons per layer.) What is the total memory capacity in terms of d-dimensional vectors?
