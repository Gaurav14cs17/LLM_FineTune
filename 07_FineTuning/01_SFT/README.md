# Supervised Fine-Tuning (SFT)

Mathematical treatment of instruction-tuning: conditional language modeling over **response tokens only**, masking rules, data formats, optimization choices, catastrophic forgetting, data mixing, and comparison to few-shot prompting.

---

## Table of Contents

- [1. Variables](#1-variables)
- [2. Intuition — What SFT Really Optimizes](#2-intuition--what-sft-really-optimizes)
- [3. Formal Objective and Masking](#3-formal-objective-and-masking)
- [3A. Cross-entropy gradients and perplexity link](#3a-cross-entropy-gradients-and-perplexity-link)
- [4. Instruction-Tuning Data Formats](#4-instruction-tuning-data-formats)
- [5. Optimization: Learning Rate and Batch Physics](#5-optimization-learning-rate-and-batch-physics)
- [6. Catastrophic Forgetting and Mitigation](#6-catastrophic-forgetting-and-mitigation)
- [7. Data Mixing Strategies](#7-data-mixing-strategies)
- [8. Full Training Pipeline (Pseudocode)](#8-full-training-pipeline-pseudocode)
- [9. SFT vs Few-Shot Prompting](#9-sft-vs-few-shot-prompting)
- [10. Numerical Examples](#10-numerical-examples)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)

---

## 1. Variables

| Symbol | Meaning |
|--------|---------|
| \(x\) | Prompt / instruction (token sequence), fixed during answer generation in training |
| \(y\) | Desired response sequence \(\{y_1,\ldots,y_T\}\) |
| \(\theta\) | Model parameters |
| \(P_\theta(y_t \mid x, y_{<t})\) | Conditional next-token probability under the LM |
| \(m_t\) | Binary mask: 0 on prompt positions, 1 on response positions |
| \(\mathcal{L}_{\mathrm{SFT}}\) | Negative log-likelihood averaged over supervised response tokens |
| \(\mathcal{D}\) | Dataset of \((x,y)\) pairs (suitably tokenized and packed) |
| \(\eta\) | Learning rate for SFT (typically \(\sim 10\times\) smaller than pretraining LR) |
| \(\pi_{\mathrm{ref}}\) | Reference policy used later in RLHF (often the SFT checkpoint) |

---

## 2. Intuition — What SFT Really Optimizes

```
     PROMPT x                          RESPONSE y
  ┌──────────────────┐             ┌────────────────────────┐
  │ "Explain logits" │  ────────► │ "Logits are unnormalised │
  │   (instruction)  │   model   │ scores before softmax..." │
  └──────────────────┘             └────────────────────────┘
           │                                    ▲
           │         TEACHER-FORCED             │  LOSS ONLY HERE
           │         (gold y_t fed as input)    │  Σ log P(y_t|...)
           ▼                                    │
  hidden states h_t carry x and y_{<t} ────────┘

  MASK:  m = [0,0,...,0 | 1,1,...,1]
              └── prompt ─┘ └── answer ──┘
```

**Plain language:** SFT is **conditional maximum likelihood** on the human (or synthetic) answer, *conditioned on the instruction*. The prompt is context, not a prediction target—so gradients do not ask the model to “re-write” the instruction; they ask it to assign high probability to each next token of the reference answer in order.

---

## 3. Formal Objective and Masking

### 3.1 Token-level cross-entropy

For a single example with response length \(|y| = T\),

\[
\mathcal{L}_{\mathrm{SFT}}(x,y;\theta)
= -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta\bigl(y_t \mid x,\, y_{<t}\bigr).
\]

Equivalently with mask \(m_t \in \{0,1\}\),

\[
\mathcal{L}_{\mathrm{SFT}}
= -\frac{\sum_t m_t \log P_\theta(w_t \mid w_{<t})}{\sum_t m_t},
\]

where \(w\) is the **concatenation** of prompt and answer in decoder-only models, and \(m_t=1\) iff \(w_t\) belongs to the supervised response span.

### 3.2 Why masking matters

- **Gradient locality:** For \(m_t=0\), \(\partial \mathcal{L}/\partial\) logits at \(t\) is **zero** under standard masked CE implementations, so parameters update only through response positions.
- **Length normalization:** Dividing by \(\sum m_t\) (not total sequence length) avoids **diluting** short answers with long prompts, which would under-train short responses.

### 3.3 Teacher forcing and covariate shift

Training uses **teacher forcing**: at position \(t\), the input is the gold prefix \(y_{<t}\). At inference, the model sees **its own** samples. Reducing train–test mismatch is partly why people tune **decoding** (temperature, top-\(p\)) separately from SFT, and why small amounts of **on-policy** data (RLHF) help.

---

## 3A. Cross-entropy gradients and perplexity link

Let \(p_\theta\) be the model’s softmax at a response position with one-hot target \(e_{y_t}\). The (unmasked) contribution is

\[
\ell_t = -\sum_k e_{y_t,k} \log p_{\theta,k}
= -\log p_{\theta,y_t}.
\]

Softmax gradient w.r.t. logits \(z\):

\[
\frac{\partial \ell_t}{\partial z_j} = p_{\theta,j} - e_{y_t,j}.
\]

Thus parameters move to **raise** mass on the true token and **lower** competitors—**probabilistic matching**.

**Perplexity:** \(\mathrm{PPL} = \exp\bigl(\frac{1}{T}\sum_{t\in \mathrm{resp}} -\log P_\theta(y_t\mid\cdot)\bigr)\). Minimising mean CE equals minimising \(\log \mathrm{PPL}\) on the supervised span—useful for side-by-side **compression-oriented** comparison across runs **when** the same tokenizer and mask conventions hold.

**Batch variance:** If microbatch responses have highly heterogeneous lengths, **weighted** accumulation by \(\sum m_t\) stabilises gradient scale versus “sum then divide by B only.”

---

## 4. Instruction-Tuning Data Formats

### 4.1 Alpaca / “instruction + input + output”

A common JSONL schema:

```json
{"instruction": "...", "input": "(optional)", "output": "..."}
```

Tokenization often wraps special **role** and **segment** markers (user / assistant). The mask leaves **instruction + input** at 0 and **output** at 1.

### 4.2 ShareGPT (multi-turn chat)

A list of turns:

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

For each assistant turn, mask only assistant spans; prior turns are context with \(m=0\) if you **do not** want to train on user text as predictions.

### 4.3 ChatML-style control tokens

Message boundaries like `<|im_start|>user`, `<|im_start|>assistant` provide unambiguous spans for masking. Mathematically it is still the same objective—**only assistant tokens participate in CE**.

---

## 5. Optimization: Learning Rate and Batch Physics

Let \(\eta_{\mathrm{PT}}\) be typical pretraining LR. Heuristic:

\[
\eta_{\mathrm{SFT}} \approx \frac{1}{10}\eta_{\mathrm{PT}}
\quad(\text{order of magnitude}).
\]

**Rationale (informal but standard):**

- Pretraining updates from **massive** web text; **signal-to-noise** per step is huge but noisy objectives dominate slowly.
- SFT data are **fewer tokens**, **sharper** gradients toward specific behaviors; a full pretraining LR often **destroys** base LM calibration and causes **catastrophic forgetting**.

Use **warmup**, **cosine decay**, or **constant LR with early stopping** depending on dataset size.

---

## 6. Catastrophic Forgetting and Mitigation

**Phenomenon:** minimizing \(\mathcal{L}_{\mathrm{SFT}}\) on a narrow distribution can reduce perplexity on **held-in** tasks while degrading **held-out** general LM abilities.

**Mitigations (often combined):**

- **Replay / mixing:** include a fraction of **general** pretraining-like text with **standard LM loss** (no masking) or with careful masking.
- **Lower LR + fewer epochs:** treat SFT as **small nudge**, not re-pretraining.
- **Regularization toward base weights** (e.g., KL to base model in later RL stages; for SFT itself, weight decay and conservative early stopping help).

---

## 7. Data Mixing Strategies

Let \(\mathcal{D}=\mathcal{D}_{\mathrm{task}}\cup\mathcal{D}_{\mathrm{aux}}\). Practical mixtures:

- **Fixed ratios:** sample mini-batches with probability \(p\) from task data and \(1-p\) from auxiliary LM data.
- **Temperatured sampling:** raise per-domain counts to power \(\alpha<1\) to **upsample** rare domains.
- **Quality filtering:** high-loss or toxic samples dominate gradients; **trim** them.

Objective for a mixed batch (schematic):

\[
\mathbb{E}_{(x,y)\sim \mathrm{Mix}}[-\sum_t m_t \log P_\theta] + \lambda \cdot \mathbb{E}_{w\sim \mathrm{LM~corpus}}[-\sum_t \log P_\theta(w_t\mid w_{<t})].
\]

The scalar \(\lambda\) trades off instruction adherence vs world knowledge retention.

---

## 8. Full Training Pipeline (Pseudocode)

```
# TOKENIZER: Add any special tokens; define which spans are "assistant".

function build_batch(rows):
    tokens, masks = [], []
    for row in rows:
        (w, m) = encode_with_mask(row)   # m_t in {0,1}
        append tokens w; append mask m
    optionally: pack multiple rows to max length with attention boundaries

function sft_step(batch, model θ, optimizer):
    logits = model(batch.tokens)                # shape [B, L, |V|]
    CE_t = CrossEntropyPerToken(logits, batch.targets)
    loss = sum_t (batch.mask * CE_t) / sum(batch.mask)
    loss.backward()
    clip_grad_norm(θ, max_norm=1.0)             # often used
    optimizer.step(); optimizer.zero_grad()
    return loss


# OUTER LOOP
θ ← load_pretrained()
for epoch in 1..E:
    for batch in DataLoader(shuffle=True):
        loss = sft_step(batch, θ, opt)
        log(loss); maybe validate_perplexity_on_dev()
save_checkpoint(θ as π_ref for later RLHF)
```

---

## 9. SFT vs Few-Shot Prompting

| Aspect | Few-shot (ICL) | SFT |
|--------|------------------|-----|
| Weight update | **None** at inference | **Yes**, offline training |
| Objective | Implicit “pattern match” in context | Explicit \(\prod_t P(y_t\mid\cdot)\) |
| Data efficiency | Excellent for many tasks at **inference** | Needs many **labeled** outputs for coverage |
| Latency / cost | Long prompts **cost tokens** | Shorter prompts possible at inference |
| Knowledge insertion | Ephemeral; hard to guarantee style | Strong style/format control |

**MAP view:** Greedy decoding seeks high-probability strings under \(P_\theta\); SFT **reshapes** \(P_\theta\) so desired behaviors inhabit **high-density** regions of that distribution.

---

## 10. Numerical Examples

### Example A — Mask and effective length

Prompt tokens: 40. Answer tokens: 10.

- **Wrong normalization (all positions):**
  \[
  \mathcal{L}_{\mathrm{bad}} = -\frac{1}{50}\sum_{\text{answer }t}\log P(y_t\mid\cdot),
  \]
  which scales gradients by \(\frac{10}{50}=0.2\) relative to “per answer token.”

- **Right normalization:**
  \[
  \mathcal{L}_{\mathrm{SFT}} = -\frac{1}{10}\sum_{\text{answer }t}\log P(y_t\mid\cdot).
  \]

So the **per-parameter update** magnitude from this example is **5×** larger with correct normalization—short answers are not drowned.

### Example B — Two-token answer

Assume:
\(P_\theta(y_1\mid x)=0.8,\; P_\theta(y_2\mid x,y_1)=0.5.\)

Joint likelihood on the answer prefix:
\[
P(y_1,y_2\mid x)=0.8\times 0.5 = 0.4.
\]

Per-token CE contribution (mean over 2):
\[
\mathcal{L}
= -\tfrac{1}{2}[\log 0.8 + \log 0.5]
\approx -\tfrac{1}{2}[-0.223 - 0.693]
\approx 0.458\;\text{nats}.
\]

### Example C — LR scaling (illustrative only)

If pretraining used \(\eta_{\mathrm{PT}}=3\times 10^{-4}\), a common SFT starting point is \(\eta_{\mathrm{SFT}}=3\times 10^{-5}\) with cosine decay; exact values depend on model size, batch tokens, and optimizer (AdamW \(\beta\)s).

---

## 11. Common Mistakes

- ❌ **Apply CE loss on user/instruction tokens** as if they were to be predicted next from scratch.

  ✓ **Mask** non-assistant spans so you train **assistant continuation**.

- ❌ **Normalize by total (prompt+answer) length** for decoder-only packed sequences without adjusting objectives.

  ✓ Normalize by **sum of mask** (response tokens).

- ❌ **Set SFT LR equal to pretraining LR** “for speed.”

  ✓ Use a **much smaller** LR and validate frequently.

- ❌ **Train many epochs** on tiny datasets.

  ✓ Early stopping; mix replay data; watch **distributional shift** metrics.

- ❌ **Assume SFT fixes all reasoning** without chain-of-thought data.

  ✓ Provide **process supervision** or complementary RL/evaluation.

---

## 12. Exercises

1. **Mask algebra:** Write \(\partial \mathcal{L}/\partial z_{t,k}\) for one response position \(t\) with one-hot target \(y_t\), softmax logits \(z_t\), and mask \(m_t\in\{0,1\}\). Show it **vanishes** when \(m_t=0\) in the standard implementation.

2. **Joint likelihood:** Prove \(\prod_{t=1}^T P(y_t\mid x,y_{<t}) = P(y_{1:T}\mid x)\) for an autoregressive model. How does this relate to “full-sequence scoring”?

3. **Format sensitivity:** Suppose a dataset accidentally leaves an assistant header inside the masked region—how does that change the entropy of targets at those tokens?

4. **Mixture coefficient:** Given two domains with per-token losses \(\ell_1,\ell_2\), derive gradient magnitude scaling when sampling mini-batches with probability \(p\) vs \((1-p)\).

5. **Calibration:** If SFT **sharpens** the answer distribution (lower entropy), how may that interact with temperature at inference?

---

### References (conceptual)

Instruction tuning and response-only losses follow standard conditional language modeling; masking conventions align with common chat-template tooling (Alpaca/ShareGPT/Chat-style formats). Always verify **your tokenizer + template** empirically—**mathematics is invariant, implementation details are not.**

