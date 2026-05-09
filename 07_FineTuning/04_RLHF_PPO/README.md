# 04 — RLHF with PPO

## 1. The RLHF Setup — Formal MDP

```
MARKOV DECISION PROCESS for language generation:

  State:      s_t = (prompt x, partial response w₁,...,wₜ₋₁)
  Action:     a_t = wₜ ∈ V  (next token to generate)
  Transition: deterministic  s_{t+1} = (x, w₁,...,wₜ)
  Reward:     r_t = 0 for t < T (intermediate steps)
              r_T = R(x, w₁,...,w_T)  (reward at episode end)
  
  (Terminal reward only — the model gets feedback at the END of generation)

POLICY:
  π_θ(wₜ | s_t) = P_θ(wₜ | x, w₁,...,wₜ₋₁)  ← the LM itself is the policy!

EPISODE:  start from BOS, generate tokens until EOS.
  Complete episode: (s₁, a₁, s₂, a₂, ..., s_T, a_T, r_T)
  
  Return: G = r_T  (since only terminal reward)

FULL RLHF OBJECTIVE:
  max_θ  J(θ) = E_{x~ρ, y~π_θ(·|x)} [R(x,y)] − β × KL(π_θ(y|x) ‖ π_ref(y|x))
  
  First term: maximise average reward
  Second term: stay near reference policy (prevent "reward hacking")
```

## 2. Bradley-Terry Reward Model — Complete Derivation

```
PREFERENCE DATA: human annotators compare (x, y_w) vs (x, y_l)
  and indicate which response is better.

BRADLEY-TERRY MODEL: model the human preference probability as:
  P(y_w ≻ y_l | x) = σ(r_φ(x, y_w) − r_φ(x, y_l))
  
  where r_φ: X × Y → ℝ is the reward model.
  σ(z) = 1/(1+e^{-z}) is the logistic sigmoid.

REWARD MODEL TRAINING (maximum likelihood on preference data):
  
  L_RM(φ) = −E_{(x,y_w,y_l)~D} [log σ(r_φ(x,y_w) − r_φ(x,y_l))]
  
  = −∑_{(x,y_w,y_l)∈D} log σ(r_φ(x,y_w) − r_φ(x,y_l))

GRADIENT:
  ∂L_RM/∂φ = −∑ (1 − σ(r_w − r_l)) × [∂r_w/∂φ − ∂r_l/∂φ]
            = ∑ σ(r_l − r_w) × [∂r_w/∂φ − ∂r_l/∂φ]
  
  correction σ(r_l−r_w):
    = 0.5 when r_w = r_l (uncertain case → large update)
    ≈ 0 when r_w >> r_l  (confident correct ranking → small update)
    ≈ 1 when r_l >> r_w  (wrong ranking → large corrective update)

REWARD MODEL ARCHITECTURE:
  Take the SFT model, add a linear head: ℝ^d → ℝ
  Input: full sequence (x, y) → pooled hidden state → scalar reward
  Training: cross-entropy on pairwise preferences
```

## 3. PPO — Full Algorithm

```
PROXIMAL POLICY OPTIMISATION (Schulman et al. 2017):

CLIPPED SURROGATE OBJECTIVE:
  J_PPO(θ) = E_t [ min( rₜ(θ) × Âₜ,  clip(rₜ(θ), 1-ε, 1+ε) × Âₜ ) ]
  
  where:
    rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)   ← probability ratio (new/old policy)
    Âₜ = advantage estimate (is this action better or worse than average?)
    ε = 0.2  (clipping threshold)

IMPORTANCE RATIO rₜ(θ):
  When rₜ(θ) > 1 + ε: new policy assigns much higher prob than old → CLIP
  When rₜ(θ) < 1 − ε: new policy assigns much lower prob than old → CLIP
  Otherwise: update normally (gradient passes through unchanged)

WHY CLIP?
  WITHOUT clipping: if Âₜ > 0 (good action), the algorithm maximises rₜ → unlimited!
    Could push rₜ very large → policy changes drastically → unstable!
  
  WITH clipping: benefit saturates at rₜ = 1+ε → bounded policy change ← stable!
  
  PROOF that PPO is lower bounded:
    min(r×A, clip(r,1-ε,1+ε)×A):
    Case A > 0:  = min(r, 1+ε) × A  ← increases only up to rₜ = 1+ε
    Case A < 0:  = max(r, 1-ε) × A  ← decreases only down to rₜ = 1-ε
    
    This ensures we don't make updates that overshoot the constraint region.
```

## 4. Advantage Estimation — GAE

```
ADVANTAGE FUNCTION:
  A(s,a) = Q(s,a) − V(s) = "how much better is action a than average?"

RETURNS (ground truth advantage):
  Gₜ = ∑_{k=t}^{T} γ^{k-t} × rₖ
  Aₜ = Gₜ − V(sₜ)  (actual return minus baseline)
  
  Problem: high variance (depends on one complete trajectory)

GENERALISED ADVANTAGE ESTIMATION (GAE, γ=1, λ=0.95):
  δₜ = rₜ + V(sₜ₊₁) − V(sₜ)    ← one-step TD error
  
  Âₜ^{GAE(γ,λ)} = ∑_{k=0}^{∞} (γλ)^k × δₜ₊ₖ
  
  Recursively: Âₜ = δₜ + (γλ) × Âₜ₊₁
  
  BIAS-VARIANCE TRADE-OFF:
    λ = 0 → Âₜ = δₜ = rₜ + V(sₜ₊₁) − V(sₜ)  (low variance, high bias — TD(0))
    λ = 1 → Âₜ = Gₜ − V(sₜ)  (no bias, high variance — Monte Carlo)
    λ = 0.95 → balanced (standard choice)

FOR RLHF (terminal reward only):
  rₜ = 0 for t < T,  r_T = reward
  
  δₜ = 0 + V(sₜ₊₁) − V(sₜ)  for t < T
  δ_T = r_T + 0 − V(s_T)      (no next state value)
  
  Âₜ = δₜ + 0.95 × δₜ₊₁ + 0.95² × δₜ₊₂ + ... + 0.95^{T-t} × δ_T
```

## 5. KL Regularisation — Mathematical Detail

```
The KL term in RLHF:
  β × KL(π_θ(y|x) ‖ π_ref(y|x))
  = β × ∑_t log[π_θ(yₜ|x,y<t) / π_ref(yₜ|x,y<t)]
  
  This is added as a PENALTY to the reward:
    r_KL = −β × log[π_θ(y_T|...) / π_ref(y_T|...)] at each step
  
  Combined reward per step:
    r_t = r_KL_t + 1[t=T] × R(x,y)
        = −β × log[π_θ(yₜ|...) / π_ref(yₜ|...)] + 1[t=T] × R(x,y)

CHOOSING β:
  SMALL β: reward maximised freely
    Risk: reward hacking (model learns to exploit reward model blindspots)
    Example: very long, repetitive outputs often fool reward models
  
  LARGE β: stays close to π_ref
    Risk: alignment doesn't improve much
    Example: generates nice English but doesn't follow instructions
  
  ADAPTIVE β: Ziegler et al. suggest adapting β to maintain:
    KL(π_θ ‖ π_ref) ≈ target_KL  (e.g., 6 nats per response)
    If KL > target: increase β (pull back toward reference)
    If KL < target: decrease β (allow more exploration)

MONITOR DURING TRAINING:
  ┌──────────────────────────────────────────────────────────────┐
  │  Healthy RLHF training metrics:                              │
  │                                                              │
  │  Reward mean:   should INCREASE over training               │
  │  KL per step:   should stay BOUNDED (not grow unboundedly)  │
  │  Loss RM:       reward model loss (should stay low)         │
  │  Value loss:    should DECREASE (critic is improving)       │
  │                                                              │
  │  Warning signs:                                              │
  │  KL growing fast → reduce β or reward hacking occurring     │
  │  Reward plateau → draft model collapses to safe boring text  │
  └──────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. Derive the gradient of the PPO clipped surrogate objective ∂J_PPO/∂θ for the case where r_t(θ) is inside the clipping range [1-ε, 1+ε].
2. Show that for terminal-only rewards (r_t=0 for t<T), the GAE advantage at step 1 is Â₁ = (γλ)^{T-1} × (R−V(s_T)) + (sum of discounted V differences). Write the closed form.
3. Compare RLHF-PPO and DPO: which requires more compute, and which is more stable? Use the number of forward passes per training step as the metric.
