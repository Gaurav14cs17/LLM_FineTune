# 01 — Supervised Fine-Tuning (SFT)

## 1. Formal Setup and Loss Function

```
DATASET:
  D = {(xᵢ, yᵢ)}_{i=1}^{N}
  xᵢ = instruction / prompt  (sequence of tokens)
  yᵢ = desired response       (sequence of tokens)

SFT OBJECTIVE:
  min_θ  L_SFT(θ) = −E_{(x,y)~D} [ ∑_{t=1}^{|y|} log P_θ(yₜ | x, y₁, ..., yₜ₋₁) ]

DIFFERENCE FROM PRETRAINING:
  Pretraining: loss over ALL tokens (including the prompt)
  SFT:         loss ONLY over response tokens y (not the prompt x)

GRADIENT EFFECT:
  ∂L_SFT/∂θ = −(1/N) ∑_{(x,y)~D} ∑_t ∂log P_θ(yₜ | x, y<t) / ∂θ
  
  The model's parameters change only to predict RESPONSE tokens better.
  Prompt tokens still influence hidden states (as context) but provide no gradient.
```

## 2. Loss Masking — Mathematical Detail

```
IMPLEMENTATION via a binary mask m_t:

  m_t = 1  if position t is a response token
      = 0  if position t is a prompt token

MASKED LOSS:
  L_SFT = − (1/∑m_t) ∑_t m_t × log P_θ(wₜ | w<t)
            ↑ normalise only by response length!

WHY NORMALISE BY RESPONSE LENGTH, NOT TOTAL LENGTH?
  Example: x = "What is 2+2?" (4 tokens), y = "4" (1 token)
  Total length = 5.  Response length = 1.
  
  Normalised by total:    loss = (1/5) × (−log P("4"|...)) 
  Normalised by response: loss = (1/1) × (−log P("4"|...))  ← 5× larger gradient!
  
  Normalising by response length ensures the model gets a full gradient signal
  even for short responses. Without this, models learn to produce short answers.

BATCH COMPUTATION:
  For batch of (x,y) pairs with different lengths, pack all into one long sequence:
  [x₁][y₁][x₂][y₂]...[xB][yB]  with mask [0..0][1..1][0..0][1..1]...
  
  Single forward pass → compute loss only at masked positions → efficient!
```

## 3. Instruction Template and its Effect

```
LLAMA-3 INSTRUCT FORMAT (complete specification):
  
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  {SYSTEM_PROMPT}
  <|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  {USER_MESSAGE}
  <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  {ASSISTANT_RESPONSE}
  <|eot_id|>

TOKENISATION AND MASK:
  Token:    [BOS] [sys_header] [system prompt tokens] [eot] [user_header] [user msg] [eot] [asst_header] [response] [eot]
  Mask:      0       0             0...0               0        0          0...0       0         0          1...1       1
             ←────────────────── no gradient ──────────────────────────────────────────────────────────────────────────→
                                                                                                ← gradient flows here →

MATHEMATICAL IMPORTANCE OF SPECIAL TOKENS:
  <|eot_id|> (end-of-turn) appears in the response portion → model learns to STOP.
  If <|eot_id|> is only in the mask-0 region: model never learns to generate it
  → model generates forever (infinite loop at inference!)
  
  CORRECT: include <|eot_id|> after the response in the mask-1 region.
```

## 4. Catastrophic Forgetting — Mathematics

```
PROBLEM: SFT on a small dataset D_SFT can overwrite pre-training knowledge.

MODEL after SFT:
  θ_SFT = θ_pretrained − η ∑_{(x,y)∈D_SFT} ∇_θ L_SFT(θ)

If D_SFT is small and concentrated in one domain:
  Parameters shift heavily toward that domain
  Pre-trained parameters for other domains are "forgotten"

MEASURING FORGETTING:
  Define: L_pretrain(θ) = loss on held-out pre-training data
  After SFT:  L_pretrain(θ_SFT) >> L_pretrain(θ_pretrained) ← forgetting!

PREVENTING FORGETTING:

Method 1 — Small learning rate:
  η_SFT << η_pretrain  (e.g., η_SFT = 1e-5 vs η_pretrain = 3e-4)
  Step size is small → parameters move little from pretrained position.

Method 2 — KL regularisation:
  L_total(θ) = L_SFT(θ) + α × KL(π_θ ‖ π_pretrained)
             = L_SFT(θ) + α × E_x[ ∑_t log[P_θ(t|x) / P_pretrained(t|x)] ]
  
  This explicitly penalises deviating from the pretrained distribution.
  α controls the trade-off: large α → less forgetting, slower alignment.

Method 3 — Few epochs (empirically effective):
  1–3 epochs on D_SFT → good alignment, minimal forgetting
  >5 epochs → overfitting on D_SFT, severe forgetting
```

## 5. Data Efficiency Analysis

```
LIMA PAPER (Zhou et al. 2023): "Less Is More for Alignment"
  1000 carefully curated (x,y) pairs → competitive with full RLHF!

MATHEMATICAL EXPLANATION:
  Pre-trained model θ already "knows" how to generate all kinds of text.
  SFT just needs to STEER the model, not teach new knowledge.
  
  Number of gradient steps:
    1000 examples × 3 epochs × (avg response 200 tokens) ÷ (batch size 32)
    = 1000 × 3 × 200 / 32 ≈ 18,750 gradient steps
  
  Vs pre-training:
    15T tokens ÷ (batch 4M tokens/step) ≈ 3,750,000 gradient steps
  
  SFT = 0.5% of pre-training compute  ← formatting/alignment is cheap!

INFORMATION THEORETIC VIEW:
  Pre-training stores vast world knowledge in θ.
  SFT conveys a SMALL amount of information:
    "respond this way, not that way" = a few bits of style/format preference.
  
  The model already has the ability — SFT unlocks and directs it.
  This is why few high-quality examples >> many noisy examples.
```

---

## Exercises

1. Why is the loss computed only on response tokens rather than the full sequence? Show mathematically that gradients from prompt tokens would not help.
2. Describe a scenario where SFT on a small biased dataset leads to catastrophic forgetting. How would KL regularisation with α=0.1 help?
3. Compute the effective training tokens per example for an instruction of 50 tokens and a response of 200 tokens. What fraction of the forward pass compute is "wasted" on the prompt that contributes no gradient?
