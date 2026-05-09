# RLHF with PPO — Reward Modeling and Policy Optimisation

End-to-end map from **SFT policy** to **preference-conditioned reward model** \(\rightarrow\) **PPO** fine-tuning. Covers the Bradley–Terry / Plackett–Luce reduction, RM loss, KL-regularised RL objective, clipped surrogate, **GAE**, value baseline, reward hacking and **full pseudocode**.

---

## Table of Contents

- [1. Variables](#1-variables)
- [2. Intuition — Three-Stage Stack](#2-intuition--three-stage-stack)
- [3. Pipeline Overview: SFT → RM → PPO](#3-pipeline-overview-sft--rm--ppo)
- [4. Bradley–Terry Preference Model](#4-bradleyterry-preference-model)
- [5. Reward Model Training Loss](#5-reward-model-training-loss)
- [6. RLHF Objective and KL Penalty](#6-rlhf-objective-and-kl-penalty)
- [7. PPO — Clipped Surrogate](#7-ppo--clipped-surrogate)
- [8. GAE (Generalized Advantage Estimation)](#8-gae-generalized-advantage-estimation)
- [8A. TE-return and finite-horizon discounting](#8a-te-return-and-finite-horizon-discounting)
- [9. Value Function Baseline](#9-value-function-baseline)
- [10. Full Algorithm — Pseudocode](#10-full-algorithm--pseudocode)
- [11. Numerical Examples](#11-numerical-examples)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)

---

## 1. Variables

| Symbol | Meaning |
|--------|---------|
| \(x\) | Prompt / dialogue prefix |
| \(y\) | Generated continuation (token sequence) |
| \(\pi_\theta\) | **Trainable** policy (causal LM) |
| \(\pi_{\mathrm{ref}}\) | **Reference** (usually SFT checkpoint) |
| \(r_\phi(x,y)\) | Scalar **reward model** (often sequence-level terminal reward) |
| \(A_t\) | **Advantage** at time \(t\) (how much better than value baseline) |
| \(V_\psi(s_t)\) | **Critic** / value network on state \(s_t\) |
| \(\beta\) | KL coefficient in \(\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\) penalty |
| \(\epsilon\) | PPO clipping hyperparameter |
| \(\gamma\) | Discount factor (often \(\approx 1\) for short horizons) |
| \(\lambda_{\mathrm{GAE}}\) | GAE \(\lambda\) for bias–variance tradeoff |

**State convention:** \(s_t = (x, y_{<t})\) with next token \(a_t = y_t\).

---

## 2. Intuition — Three-Stage Stack

```
Stage 1: SFT                    π_ref  (imitation on demonstrations)
              │
              ▼
Stage 2: Reward Model r_φ       learns scalar scores from (chosen, rejected) pairs
              │
              ▼
Stage 3: PPO on π_θ             max E[r] − β KL(π_θ || π_ref)
              │
              └── KL keeps style/safety near reference while reward pulls "better" behaviors
```

**Why not stop at SFT?** Demonstrations are **finite** and often **suboptimal**. Preference labels let the model trade off competing desiderata via a **learned scalar** aligned to humans.

---

## 3. Pipeline Overview: SFT → RM → PPO

1. **SFT:** supervised finetune \(\pi_{\mathrm{ref}}\) to imitate high-quality answers.
2. **Preference collection:** for each \(x\), sample pairs \((y_w, y_l)\); humans mark \(y_w \succ y_l\).
3. **RM training:** fit \(r_\phi\) by maximizing preference likelihood (Bradley–Terry).
4. **PPO:** treat LM as policy; optimize surrogate with advantage from **reward + KL** shaping; often **add** KL penalty **per token/batch** to \(\pi_{\mathrm{ref}}\).

---

## 4. Bradley–Terry Preference Model

For two competing completions given the same context:

\[
\mathbb{P}\bigl(y_w \succ y_l \mid x\bigr)
= \sigma\!\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr),
\qquad \sigma(z) = \frac{1}{1+e^{-z}}.
\]

**Interpretation:** The **difference** of scalar scores estimates the log-odds of preference:

\[
\log \frac{\mathbb{P}(y_w \succ y_l)}{\mathbb{P}(y_l \succ y_w)}
= r_\phi(x,y_w) - r_\phi(x,y_l).
\]

This is the **Bradley–Terry** link; extensions to **rankings** use Plackett–Luce style objectives.

---

## 5. Reward Model Training Loss

Given dataset \(\mathcal{D}=\{(x, y_w^{(i)}, y_l^{(i)})\}\),

\[
\mathcal{L}_{\mathrm{RM}}(\phi)
= -\mathbb{E}_{\mathcal{D}}\Bigl[
\log \sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr)
\Bigr].
\]

**Gradient sketch:** Let \(\Delta = r_\phi(x,y_w)-r_\phi(x,y_l)\).

\[
\frac{\partial \mathcal{L}_{\mathrm{RM}}}{\partial \phi}
= -\bigl(1-\sigma(\Delta)\bigr)\,
\Bigl(\nabla_\phi r_\phi(x,y_w) - \nabla_\phi r_\phi(x,y_l)\Bigr).
\]

**Margin intuition:** When \(\Delta\) is large and positive (already confident correct), \(\sigma(\Delta)\approx 1\) and gradient **vanishes**—the model stops overfitting **easy** pairs; hard pairs still train.

---

## 6. RLHF Objective and KL Penalty

**Composite reward** often used in practice (schematic):

\[
R_{\mathrm{total}}(x,y)
= r_\phi(x,y)
- \beta_{\mathrm{KL}}\,
\mathbb{E}_{t}\!\left[
\log \frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}
\right],
\]

where the expectation over \(t\) can be implemented as **per-token** penalty summed along the sequence, or batch-averaged variants.

**Why KL?** The policy \(\pi_\theta\) can **game** \(r_\phi\) (exploit spurious correlates). Staying close to \(\pi_{\mathrm{ref}}\) limits **distribution shift**.

---

## 7. PPO — Clipped Surrogate

Let **importance ratio**

\[
\rho_t(\theta)
= \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]

**Surrogate (single step):**

\[
L^{\mathrm{CLIP}}_t(\theta)
= \min\Bigl(
\rho_t(\theta)\, \hat{A}_t,\;
\mathrm{clip}(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t
\Bigr).
\]

**Objective to maximise (averaged):**

\[
J_{\mathrm{PPO}}(\theta)
= \mathbb{E}_t\bigl[L^{\mathrm{CLIP}}_t(\theta)\bigr]
- c_V \cdot \mathbb{E}_t\bigl[(V_\psi(s_t)-\hat{G}_t)^2\bigr]
\;+\; c_E \cdot \mathcal{H}(\pi_\theta)
\]

(common **value** and **entropy bonus** terms; coefficients depend on codebase).

**Clip effect:** Large policy updates that overshoot advantage signs get **flattened**, improving monotonic improvement heuristics from TRPO lineage.

---

## 8. GAE (Generalized Advantage Estimation)

Define **temporal-difference residual** (scalar reward \(R_t\)—often KL-shaped per token or terminal RM):

\[
\delta_t = R_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t).
\]

**GAE(\(\lambda\)) advantage:**

\[
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
= \sum_{l=0}^{\infty} (\gamma\lambda_{\mathrm{GAE}})^l\, \delta_{t+l}.
\]

**Bias–variance:**

- \(\lambda_{\mathrm{GAE}}=0\): one-step TD (low variance, biased).
- \(\lambda_{\mathrm{GAE}}=1\): Monte Carlo-ish return path (higher variance).

In LM generation, horizons are long but **Markovian** assumption is approximate due to **token-level coupling**—still widely used with careful reward shaping.

---

## 8A. \(T_E\)-return and finite-horizon discounting

Let the **undiscounted** finite-horizon return from time \(t\) be \(G_t=\sum_{l=0}^{T-t} R_{t+l}\). With discount \(\gamma\in(0,1]\),

\[
G_t^{(\gamma)} = \sum_{l=0}^{\infty} \gamma^l R_{t+l}.
\]

**GAE** mixes **multi-step** bootstraps:

\[
\hat{A}_t = \sum_{l\ge 0} (\gamma\lambda_{\mathrm{GAE}})^l \delta_{t+l},
\quad
\delta_{t+l}=R_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l}).
\]

For **language**, \(R_{t+l}\) often mixes **dense KL penalties** with **sparse** RM at EOS—then \(\delta\) encodes **how surprising** the actual token was **relative** to value baseline **plus** immediate shaping.

**Practical:** Many codebases **center** advantages \(\hat{A}_t \leftarrow (\hat{A}_t-\mu)/\sigma\) across minibatches for stable PPO.

---

## 9. Value Function Baseline

**Advantage** centers the return:

\[
\hat{A}_t \approx \bigl(\sum_{l\ge 0} \gamma^l R_{t+l}\bigr) - V_\psi(s_t).
\]

Training \(V_\psi\) by regression to **returns** (or bootstrapped \(\delta\)) **reduces variance** of policy gradients.

**Sequence-level RM:** Often \(r_\phi\) is only at **EOS**—then intermediate \(R_t\) may be mostly **shaping / KL penalty**, with terminal **sparse** reward, mirroring bandit-like credit assignment; **reward normalization / whitening** across batch is common.

---

## 10. Full Algorithm — Pseudocode

```
# INPUT: π_ref (SFT LM), preference data for r_φ
train_reward_model(r_φ, preference_batches):
    for (x, y_w, y_l) in preference_batches:
        loss = −log σ( r_φ(x,y_w) − r_φ(x,y_l) )
        SGD step on φ

initialize π_θ ← π_ref  (same architecture / weights copy)
initialize critic V_ψ
for PPO_iteration in 1..K:
    collect rollouts:
        for prompts x in batch:
            sample y ∼ π_{θ_old}(·|x)
            compute per-token log π_ref, log π_θ_old, KL penalties
            compute scalar rewards:
               R_terminal = r_φ(x,y)
               distribute/shape per-token R_t per recipe
            store trajectories

    advantages Â ← GAE({R_t}, {V_ψ}, γ, λ_GAE)

    for minibatch in shuffled(rollouts):
        ρ_t = π_θ / π_{θ_old} on stored actions
        L_clip ← clipped surrogate with Â
        L_vf ← (V_ψ − V_target)^2
        loss ← −(L_clip) + c_vf L_vf − c_H Entropy(π_θ) + β_KL·KL_terms
        update θ, ψ with Adam (multiple epochs per batch in classic PPO)

return π_θ
```

**Practical notes:**

- Store **old** logprobs at collection time.
- **Reject sampling / overlong sequences** handling interacts with advantage math.

---

## 11. Numerical Examples

### Example 1 — Bradley–Terry probability

If \(r_\phi(x,y_w)=2.0\) and \(r_\phi(x,y_l)=0.5\), then \(\Delta=1.5\).

\[
\sigma(1.5) = \frac{1}{1+e^{-1.5}} \approx 0.818.
\]

So the model assigns \(\approx 81.8\%\) probability that \(y_w\) beats \(y_l\).

### Example 2 — KL per token (toy)

Suppose at some step \(\log \pi_\theta - \log \pi_{\mathrm{ref}} = 0.1\) nats for chosen action.

Over 128 generated tokens, **if** this were constant per token, cumulative \(\sum \approx 12.8\) nats (actual KL is expectation-dependent; this is only a **scalable intuition**).

### Example 3 — PPO clipping

Let \(\hat{A}_t=+2\), \(\rho_t=1.3\), \(\epsilon=0.2\).

- Unclipped term \(= 1.3\times 2 = 2.6\).
- Clipped \(\rho\) \(= \mathrm{clip}(1.3,0.8,1.2)=1.2\), clipped term \(= 2.4\).

Take **min** \(\Rightarrow 2.4\) stops overly optimistic importance ratio boosts.

---

## 12. Common Mistakes

- ❌ **Train RM with single-label classification without pairwise structure.**

  ✓ Pairwise **log-sigmoid** aligns scales up to monotonic transform.

- ❌ **Use unconstrained PPO on LM without KL to \(\pi_{\mathrm{ref}}\)** when RM is weak.

  ✓ KL + conservative \(\beta\) schedule reduces **reward hacking**.

- ❌ **Provide only terminal RM on EOS but forget per-step baselines.**

  ✓ Implement **GAE consistently** with how \(R_t\) is defined.

- ❌ **Reuse stale \(\pi_{\theta_{\mathrm{old}}}\) for many steps without recollection.**

  ✓ Refresh rollouts; cap **PPO epochs** like classic PPO.

- ❌ **Confuse \(\sigma(\Delta)\) with raw reward gap.**

  ✓ Optimize **log-likelihood of preferences**, not raw MSE on scores.

---

## 13. Exercises

1. **Show** \(\partial \mathcal{L}_{\mathrm{RM}}/\partial r_w\) when \(r_w\) appears positively inside \(\sigma(\cdot)\).

2. **Derive** the Monte Carlo policy gradient with **baseline** and prove adding \(b(s)\) with \(\mathbb{E}[b|s]\) constant w.r.t. action choice does not bias the gradient (under correctness assumptions).

3. **Relate** PPO clip to **trust region**: why does clipping penalize huge \(\rho_t\) when \(\hat{A}_t>0\)?

4. **Sparse RM:** Write \(\delta_t\) when \(R_t=0\) for \(t<T\) and \(R_T = r_\phi(x,y)\).

5. **KL estimation:** Compare **k-sample** KL estimators vs closed-form for Gaussians—when are they fragile for LMs?

---

### Short synthesis

RLHF-PPO is not “magic alignment”—it is **preference inverse reinforcement learning** paired with **trust-region policy optimisation** on gigantic policies. Success depends on **reward quality**, **KL anchoring**, and **honest evaluation** outside the training preference distribution.
