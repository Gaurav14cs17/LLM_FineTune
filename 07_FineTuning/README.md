# Chapter 07 — Fine-tuning

Base LMs optimise **next-token likelihood**; fine-tuning **specialises** behaviour toward instructions, safety, or domains without always retraining from scratch. This chapter covers **SFT**, efficient adapters (**LoRA**, **QLoRA**), and alignment objectives (**RLHF/PPO**, **DPO**) with the gradients and reparameterisations you need to implement or debug them.

```
   base weights W_*  (frozen or slow)
           │
   ┌───────┴────────┐
   │  SFT: train all or last layers                                  │
   │  LoRA: W = W0 + B·A   (A∈ℝ^{r×k}, B∈ℝ^{d×r}, r≪d)           │
   │  Preferences: pairwise losses on (x, y_w, y_l)                  │
   └─────────────────────────────────────────────────────────────────┘
```

## Sub-chapters

**`01_SFT/` (Supervised Fine-Tuning)**  
Trains on demonstration tuples \((x,y)\) with teacher forcing; masks **prompt** vs **response** tokens in the loss; discusses data mixture ratios and **epoch** sizing when the base model is already strong.

**`02_LoRA/` (Low-Rank Adaptation)**  
Adds trainable low-rank deltas to selected projections; counts parameters \(r(d+k)\) vs full \(d\cdot k\); rank–performance curves; where to attach adapters (attention vs FFN) in practice.

**`03_QLoRA/`**  
Base weights in **4-bit NF4** (or similar); adapters in higher precision; memory arithmetic for viable single-GPU tuning of large bases; caveats on quantisation noise vs rank.

**`04_RLHF_PPO/`**  
Reward model \(r_\phi\) trained on comparisons; policy \(\pi_\theta\) fine-tuned with **PPO** / proximal objectives; KL penalty toward reference \(\pi_{\mathrm{ref}}\) to avoid collapse; advantage estimation basics (`GAE` at high level).

**`05_DPO/` (Direct Preference Optimisation)**  
Rewrites the Bradley–Terry / RLHF objective to **pure supervised** loss on preferences—no explicit reward model at run time in the minimal variant; stable offline training from pairwise data.

## Key formulas and concepts

**SFT loss (response tokens only)**

\[
\mathcal{L}_{\mathrm{SFT}} = -\sum_{t\in\mathcal{R}} \log \pi_\theta(y_t\mid x, y_{<t}).
\]

**LoRA reparameterisation**

\[
W = W_0 + B A,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k}.
\]

**DPO (binary pref; schematic)**

\[
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\Bigl[\log \sigma\Bigl(\beta\bigl(\log \tfrac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \log \tfrac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\bigr)\Bigr)\Bigr].
\]

**PPO clip (reminder)**

\[
L_{\mathrm{CLIP}} = -\mathbb{E}_t\bigl[\min(r_t \hat{A}_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\bigr].
\]

## Prerequisites

- Chapters 01 and 06 (losses, optimisation).
- Chapter 05 (where LoRA attaches inside blocks).

## Safety / evaluation hook

Fine-tuning can **overwrite** alignment gains—watch held-out adversarial prompts and retain a frozen reference \(\pi_{\mathrm{ref}}\) in KL-regularised setups.

## Study tips

- Compare **trainable parameter counts** for LoRA on all \(W_Q,W_K,W_V,W_O\) vs only \(W_Q,W_V\) variants cited in papers.
- Implement DPO on a tiny GPT; verify loss decreases when \(\pi_\theta\) puts more mass on \(y_w\) than \(y_l\).

## Data hygiene

- Avoid **prompt leakage** from evaluation sets into SFT mixtures; even small contamination inflates instruction-following metrics.
- Keep **base-model capability** frozen via KL to \(\pi_{\mathrm{ref}}\) when iterative RLHF risks **reward hacking** (model exploits proxy reward).

## Resource planning

- QLoRA trades **adapter VRAM** for slower **quantised base** forward; beneficial when single-GPU fine-tune of a 70B-class base is the goal.

## Evaluation stack

- Track **SFT loss** on *held-out* demonstrations distinct from RLHF prompts to catch **overfitting / memorisation** (model outputs training IDs verbatim).

## Intended outcomes

You should be able to: (1) specify which tensors receive gradients in LoRA vs full fine-tune; (2) write LoRA FLOPs / memory scaling in big-\(O\) vs rank \(r\); (3) contrast offline DPO vs online PPO from a data and infra standpoint; (4) diagnose **KL blow-up** in RLHF when the reward is misspecified.

## Exam-style checks

- For matrices \(W\in\mathbb{R}^{4096\times 4096}\), compare trainable entries in full FT vs LoRA rank \(r=16\) on this single matrix.
- Describe a failure mode where DPO improves **preference win-rate** while **degrading** perplexity on general corpora—when might you still accept that trade?

**Mini extension:** count trainable scalars for **LoRA on \(W_Q\) and \(W_V\) only** vs **all four** projection types inside one self-attention block, keeping rank \(r\) shared.

Align adapter **initialization** (e.g. \(A\) Gaussian, \(B=0\)) with why early training steps resemble the frozen-base forward map.

---

_End of chapter overview._
