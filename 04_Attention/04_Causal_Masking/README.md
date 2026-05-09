# 04 — Causal Masking

## 1. The Problem: Attention Sees the Future

```
WITHOUT causal mask:

  Sequence: ["The", "cat", "sat", "on"]
  
  When predicting "sat" (position 2), the model CAN see "on" (position 3):
  
  Token "sat" attends to:
  ┌───────────────────────────────────────────┐
  │  "The" (past) ✓  attention weight: 0.2   │
  │  "cat" (past) ✓  attention weight: 0.5   │
  │  "sat" (self) ✓  attention weight: 0.2   │
  │  "on"  (FUTURE!) attention weight: 0.1   │  ← LEAKS future token!
  └───────────────────────────────────────────┘
  
  Training: Model sees "on" to predict "sat" → trivially correct → useless
  Inference: "on" doesn't exist yet → can't use it → INCONSISTENCY
```

## 2. The Causal Mask

```
Mask M for L=5 (0=allowed, -∞=blocked):

      KEY POSITIONS →
      k=0   k=1   k=2   k=3   k=4
    ┌──────────────────────────────────┐
q=0 │   0     -∞    -∞    -∞    -∞   │  BOS sees only itself
    ├──────────────────────────────────┤
q=1 │   0      0    -∞    -∞    -∞   │  token 1 sees tokens 0,1
    ├──────────────────────────────────┤
q=2 │   0      0     0    -∞    -∞   │  token 2 sees tokens 0,1,2
    ├──────────────────────────────────┤
q=3 │   0      0     0     0    -∞   │  token 3 sees tokens 0,1,2,3
    ├──────────────────────────────────┤
q=4 │   0      0     0     0     0   │  token 4 sees all tokens
    └──────────────────────────────────┘
    Lower triangular = ✓ allowed
    Upper triangular = -∞ → exp(-∞)=0 → 0 weight after softmax

Implementation:
  S_masked = S + M
  where M[i,j] = 0 if j≤i else -1e9 (or float('-inf'))
  A = softmax(S_masked)   → upper triangle = 0 automatically
```

## 3. Teacher Forcing — Training in Parallel

```
Without masking: must generate token-by-token (L separate forward passes)
With causal mask: process ALL L tokens in ONE forward pass!

Input:  [BOS, "The", "cat", "sat", "on"]
        ─────────────────────────────────
Target: ["The", "cat", "sat", "on", EOS]

ONE forward pass, 5 parallel predictions:
  Position 0: context=[BOS]            → predict "The"
  Position 1: context=[BOS,"The"]      → predict "cat"
  Position 2: context=[BOS,"The","cat"] → predict "sat"
  Position 3: context=[BOS,"The","cat","sat"] → predict "on"
  Position 4: context=[BOS,"The","cat","sat","on"] → predict EOS

CAUSAL MASK ENSURES: position 2 ONLY sees positions 0,1,2 — NOT 3,4

Speedup:
  Without masking: L sequential steps
  With masking:    1 parallel step = L× faster training!

This is the CORE efficiency advantage of the Transformer over RNNs.
```

## 4. Why exp(-∞) = 0 Works

```
Masked softmax computation:

  Score vector (row i): [s₁, s₂, -∞, -∞]
  
  exp([s₁, s₂, -∞, -∞]) = [e^s₁, e^s₂, 0, 0]
  
  sum = e^s₁ + e^s₂ + 0 + 0
  
  softmax = [e^s₁/(e^s₁+e^s₂), e^s₂/(e^s₁+e^s₂), 0, 0]
             ↑ past tokens share probability     ↑ future = exactly 0

In PyTorch:
  torch.finfo(torch.float32).min  ≈ -3.4e38 (approximation for -∞)
  In BF16: use -1e4 (overflow-safe)
```

---

## Exercises

1. Write out the 4×4 causal mask matrix M for L=4.
2. Prove that with teacher forcing and causal masking, training a Transformer of L tokens is equivalent to training L separate prefix-based language models simultaneously.
3. Why would removing the causal mask (using bidirectional attention) during inference of a decoder-only model give incorrect results?
