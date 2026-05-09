# 03 — Entropy and Information Theory

## 1. Shannon Entropy — Formal Definition and Derivation

```
QUESTION: What is a natural measure of "uncertainty" or "information content"?

AXIOMS (uniquely determine entropy up to a constant):
  (A1) H is continuous in all p_i
  (A2) For uniform distribution over n outcomes: H increases with n
  (A3) H is consistent: grouping outcomes doesn't change total H
       H(p₁, p₂, p₃) = H(p₁+p₂, p₃) + (p₁+p₂)·H(p₁/(p₁+p₂), p₂/(p₁+p₂))

THEOREM (Shannon 1948): The unique function (up to scaling) satisfying A1-A3 is:
  H(X) = −∑_{x ∈ X} P(x) · log P(x)

PROOF SKETCH:
  For uniform P over n outcomes: H = K log n for some constant K.
  Setting K=1 (nats) or K=1/log₂ (bits) gives the formula.
  Non-uniform case follows from axiom A3.

UNITS:
  log = natural log (ln): H measured in NATS
  log = log₂:             H measured in BITS
  Conversion: 1 nat = log₂(e) ≈ 1.4427 bits

EXAMPLES:
  Deterministic (one outcome has P=1):
    H = −(1·log 1 + 0·log 0) = −(0 + 0) = 0 nats  (0·log0 = 0 by convention: lim_{x→0} x ln x = 0)
  
  Fair coin flip (P("H")=P("T")=0.5):
    H = −(0.5·log 0.5 + 0.5·log 0.5) = −log(0.5) = log(2) ≈ 0.693 nats = 1 bit
  
  Uniform over |V|=128000 tokens:
    H = −∑ (1/128000) · log(1/128000) = log(128000) ≈ 11.76 nats ≈ 16.96 bits
  
  Natural language (empirically): H ≈ 0.7-1.5 nats/token
```

## 2. Maximum Entropy Theorem — Formal Proof

```
THEOREM: Among all distributions over n outcomes, the UNIFORM distribution maximises H.

PROOF using Lagrange multipliers:
  Maximise:  H = −∑_i pᵢ log pᵢ
  Subject to: ∑_i pᵢ = 1 (normalisation constraint)

  Lagrangian:
    L = −∑_i pᵢ log pᵢ − λ(∑_i pᵢ − 1)

  First-order condition:
    ∂L/∂pᵢ = −log pᵢ − 1 − λ = 0  for all i
    ⟹ log pᵢ = −1 − λ = const    for all i
    ⟹ pᵢ = e^{−1−λ} = const      for all i

  So optimal pᵢ is IDENTICAL for all i = 1/n. ■

MATHEMATICAL PROOF OF H_max = log n:
  H_uniform = −∑_{i=1}^{n} (1/n) · log(1/n)
             = −n × (1/n) · log(1/n)
             = −log(1/n) = log(n)

FOR LLMs: H_max = log(|V|)
  If model output is uniform: H = log(128000) ≈ 11.76 nats → PPL = 128000 (terrible)
  If model assigns all weight to correct token: H = 0 → PPL = 1 (perfect)
```

## 3. KL Divergence — Full Properties and Proof of Non-Negativity

```
DEFINITION:
  KL(P ‖ Q) = ∑_x P(x) · log[P(x)/Q(x)]
            = ∑_x P(x) · log P(x) − ∑_x P(x) · log Q(x)
            = −H(P) + H(P,Q)
            = H(P,Q) − H(P)

  where H(P,Q) = −∑_x P(x)·log Q(x) is the cross-entropy.

THEOREM (Gibbs' inequality): KL(P ‖ Q) ≥ 0, with equality iff P = Q.

PROOF using Jensen's inequality:
  KL(P ‖ Q) = −∑_x P(x) · log[Q(x)/P(x)]
            ≥ −log ∑_x P(x) · [Q(x)/P(x)]    ← Jensen: −log is convex
            = −log ∑_x Q(x)
            = −log 1
            = 0   ■

  Equality holds iff Q(x)/P(x) = const for all x, which (given ∑P=∑Q=1) means P=Q.

INTUITION:
  KL(P ‖ Q) = "how many extra nats/bits needed to encode P-distributed data using Q-code"
  = information lost when using Q instead of P

ASYMMETRY:
  KL(P ‖ Q) ≠ KL(Q ‖ P) in general.
  
  EXAMPLE: P = [0.9, 0.1], Q = [0.5, 0.5]
  
  KL(P‖Q) = 0.9·log(0.9/0.5) + 0.1·log(0.1/0.5)
           = 0.9·log(1.8) + 0.1·log(0.2)
           = 0.9×0.588 + 0.1×(−1.609) = 0.529 − 0.161 = 0.368 nats
  
  KL(Q‖P) = 0.5·log(0.5/0.9) + 0.5·log(0.5/0.1)
           = 0.5·(−0.588) + 0.5·(1.609) = −0.294 + 0.805 = 0.511 nats
  
  KL(P‖Q)=0.368 ≠ KL(Q‖P)=0.511 → asymmetric!
```

## 4. Cross-Entropy Loss = KL Minimisation

```
TRAINING OBJECTIVE: L(θ) = E_{x~data}[−log P_θ(x)]

This equals the CROSS-ENTROPY H(P_data, P_θ):
  H(P_data, P_θ) = −∑_x P_data(x) · log P_θ(x)
                 = −E_{x~P_data}[log P_θ(x)]
                 = L(θ)   ✓

Now use the identity H(P, Q) = H(P) + KL(P ‖ Q):
  L(θ) = H(P_data, P_θ) = H(P_data) + KL(P_data ‖ P_θ)
                              ↑               ↑
                           constant in θ    depends on θ

THEREFORE:
  min_θ L(θ) = min_θ KL(P_data ‖ P_θ)   (since H(P_data) is constant w.r.t. θ)

  TRAINING = MINIMISING KL DIVERGENCE from the data distribution to the model!

WHAT DOES THIS ACHIEVE?
  KL(P_data ‖ P_θ) → 0 means P_θ → P_data
  The model learns to match the training data distribution.
  
  LIMITATION: KL(P_data ‖ P_θ) is zero-forcing (forces P_θ to cover all modes of P_data).
  If P_data has rare modes, the model is penalised for assigning low probability to them.
  This causes LLMs to be "conservative" in their predictions.
```

## 5. Mutual Information — LLM Perspective

```
DEFINITION:
  I(X; Y) = H(X) + H(Y) − H(X,Y)
           = KL(P(X,Y) ‖ P(X)·P(Y))  ← KL from joint to product of marginals
           ≥ 0   (by Gibbs' inequality applied to this KL)

INTERPRETATIONS:
  I(X;Y) = H(X) − H(X|Y)   (reduction in uncertainty about X given Y)
  I(X;Y) = H(Y) − H(Y|X)   (reduction in uncertainty about Y given X)
  
  I(X;Y) = 0 iff X and Y are independent (knowing Y gives no info about X).

ATTENTION AS MUTUAL INFORMATION:
  Head h measures something related to I(token_i ; token_j)
  High attention A_ij between positions i,j indicates that position j's content
  is mutually informative with position i → they should share information.
  
  Different attention heads specialise in different conditional MI:
    Head 1: I(token_i ; token_j | syntactic_role)  ← syntax
    Head 2: I(token_i ; token_j | coreference)     ← pronouns
    Head 3: I(token_i ; token_j | semantic_class)  ← meaning

INFORMATION BOTTLENECK INTERPRETATION OF LAYERS:
  Each layer compresses the input while preserving task-relevant information.
  
  Layer l: X → Z_l → Y
  Goal:  min I(X; Z_l) − I(Z_l; Y)   (compress X while retaining Y-relevant info)
  
  As l increases, Z_l becomes a more compressed and task-relevant representation.
  Early layers: retain most of X (low compression)
  Later layers: retain mainly task-relevant aspects (high compression)
```

---

## Exercises

1. Compute H(X) for X distributed as P(1)=0.5, P(2)=0.25, P(3)=0.125, P(4)=0.125. In bits.
2. Prove KL(P‖Q) ≥ 0 using Jensen's inequality applied to the function −log(x) (which is convex). Show all steps clearly.
3. Show mathematically that minimising cross-entropy H(P_data, P_θ) with respect to θ is equivalent to minimising KL(P_data ‖ P_θ). This is the fundamental link between information theory and machine learning.
