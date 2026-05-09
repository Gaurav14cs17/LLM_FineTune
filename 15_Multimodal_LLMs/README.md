# Understanding Multimodal LLMs

*From CLIP to LLaVA — connecting vision, language, and beyond*

---

**Multimodal LLMs** extend language models to understand images, audio, video, and other modalities. Models like GPT-4V, LLaVA, and Gemini process visual inputs alongside text, enabling image understanding, visual reasoning, and document analysis.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What is Multimodal?](#11-what-is-multimodal)
   - [1.2 Architecture Patterns](#12-architecture-patterns)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Vision Encoders](#2-vision-encoders)
   - [2.1 CLIP (Contrastive Language-Image Pretraining)](#21-clip)
   - [2.2 ViT (Vision Transformer)](#22-vit-vision-transformer)
3. [Vision-Language Alignment](#3-vision-language-alignment)
   - [3.1 The Projection Layer](#31-the-projection-layer)
   - [3.2 Training Stages](#32-training-stages)
4. [LLaVA Architecture](#4-llava-architecture)
   - [4.1 Image Tokenization](#41-image-tokenization)
   - [4.2 Interleaved Image-Text](#42-interleaved-image-text)
5. [Summary](#5-summary)
   - [5.1 Formulas Quick Reference](#51-formulas-quick-reference)
   - [5.2 Common Mistakes](#52-common-mistakes)
6. [Exercises](#6-exercises)

---

## 1. Overview

### 1.1 What is Multimodal?

```
┌─────────────────────────────────────────────────────────────┐
│  TEXT-ONLY LLM:                                             │
│  Input: "Describe this image" → Output: ??? (no image!)    │
│                                                             │
│  MULTIMODAL LLM:                                            │
│  Input: [IMAGE] + "Describe this image"                     │
│  Output: "A cat sitting on a windowsill, looking outside    │
│           at birds. The cat is orange with white stripes..." │
│                                                             │
│  The model can SEE + READ + REASON about visual content!   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture Patterns

```
PATTERN A: ENCODER + PROJECTOR + LLM (LLaVA, InternVL)
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  Vision   │───►│Projector │───►│   LLM    │───► text
  │  Encoder  │    │ (MLP)   │    │(decoder) │
  └──────────┘    └──────────┘    └──────────┘
       ↑                               ↑
     image                            text tokens

PATTERN B: EARLY FUSION (Gemini, GPT-4V)
  Image patches are tokenized and fed DIRECTLY into transformer
  alongside text tokens. No separate vision encoder.
  
  [img_patch_1, img_patch_2, ..., img_patch_N, text_tok_1, ...]
  All processed by the same transformer.

PATTERN C: CROSS-ATTENTION (Flamingo)
  Image features injected via CROSS-ATTENTION layers
  interleaved between self-attention layers.
  
  Text tokens attend to image features via cross-attention,
  but image features don't attend to text (asymmetric).
```

### 1.3 Pipeline Summary

```
IMAGE PROCESSING for LLaVA-style model:

  Raw image (e.g., 336×336 pixels)
      ↓
  Split into patches (14×14 pixels each = 576 patches)
      ↓
  Vision encoder (CLIP ViT-L/14) → 576 visual tokens ∈ ℝ^{1024}
      ↓
  Projector (2-layer MLP) → 576 tokens ∈ ℝ^{4096} (LLM dimension)
      ↓
  Concatenate with text tokens:
  [<img>, v₁, v₂, ..., v₅₇₆, </img>, "Describe", "this", "image"]
      ↓
  LLM processes all tokens (visual + text) with standard attention
      ↓
  Generate response autoregressively
```

---

## 2. Vision Encoders

### 2.1 CLIP

```
CLIP (Radford et al. 2021):
  Trained on 400M image-text pairs from the internet.
  
  TRAINING OBJECTIVE (contrastive):
    For batch of N (image, text) pairs:
    
    sim(I_i, T_j) = cos(f_img(I_i), f_txt(T_j))
    
    L = -(1/N) × Σᵢ [log exp(sim(I_i,T_i)/τ) / Σⱼ exp(sim(I_i,T_j)/τ)]
    
    Maximize similarity of MATCHING pairs (diagonal).
    Minimize similarity of NON-matching pairs (off-diagonal).

  RESULT: image and text embeddings in SAME vector space.
    "a photo of a cat" → embedding close to cat images
    "a photo of a dog" → embedding close to dog images

WHAT MAKES CLIP USEFUL FOR MULTIMODAL LLMs:
  CLIP's visual features are already SEMANTICALLY aligned with language.
  A cat image produces features that are naturally close to "cat" text.
  → Much easier for the LLM to understand visual features!
```

### 2.2 ViT (Vision Transformer)

```
ViT processes images as sequences of patches:

  Image: H × W × 3 (e.g., 336 × 336 × 3)
      ↓
  Split into P × P patches (P=14): 24 × 24 = 576 patches
      ↓
  Each patch: 14 × 14 × 3 = 588 pixels → linear projection → d_vis dims
      ↓
  Add position embeddings (learned, one per patch)
      ↓
  Process through L_vis transformer layers
      ↓
  Output: 576 patch tokens × d_vis (1024 for ViT-L)

MATH:
  x_patches ∈ ℝ^{N_patches × (P² × 3)}   N_patches = (H/P) × (W/P)
  
  z₀ = x_patches × W_embed + pos_embed    W_embed ∈ ℝ^{(P²×3) × d_vis}
  z_L = ViT_Transformer(z₀)               z_L ∈ ℝ^{N_patches × d_vis}

RESOLUTION SCALING:
  Higher resolution → more patches → more visual tokens:
    224×224 @ P=14: 256 patches (standard)
    336×336 @ P=14: 576 patches (LLaVA-1.5)
    448×448 @ P=14: 1024 patches (high-res)
    672×672 @ P=14: 2304 patches (very high-res)
  
  More patches = more detail but more tokens in LLM context!
```

---

## 3. Vision-Language Alignment

### 3.1 The Projection Layer

```
PROBLEM: Vision encoder outputs d_vis = 1024 dimensional tokens.
         LLM expects d_llm = 4096 dimensional tokens.
         We need to bridge these two spaces.

PROJECTION OPTIONS:
  Linear:    W_proj ∈ ℝ^{d_vis × d_llm}  (simple, fast)
  2-layer MLP: x → Linear(d_vis, d_llm) → GELU → Linear(d_llm, d_llm)
  Cross-attention (Q-Former): learned queries attend to vision features

LLaVA uses 2-layer MLP:
  v_projected = GELU(v_visual × W₁ + b₁) × W₂ + b₂
  
  W₁ ∈ ℝ^{1024 × 4096}, W₂ ∈ ℝ^{4096 × 4096}
  Parameters: ~20M (tiny compared to 8B LLM)
```

### 3.2 Training Stages

```
LLAVA TWO-STAGE TRAINING:

STAGE 1: PRETRAINING (alignment)
  - Freeze: vision encoder AND LLM
  - Train: ONLY the projection layer
  - Data: 600K image-caption pairs (CC3M subset)
  - Goal: teach projector to map visual features into LLM's "language"
  - Duration: ~1 epoch, a few hours on 8 GPUs

STAGE 2: FINE-TUNING (instruction following)
  - Freeze: vision encoder
  - Train: projection layer AND LLM
  - Data: 665K visual instruction-following examples
  - Goal: teach the full model to follow visual instructions
  - Duration: ~1 epoch, ~1 day on 8 GPUs

WHY TWO STAGES?
  Stage 1 (projection only): cheap, establishes basic alignment.
    If you train LLM directly from random projection, it gets confused.
  Stage 2 (LLM unfrozen): fine-tunes the LLM to USE visual features.
    LLM learns to reason about what it "sees."
```

---

## 4. LLaVA Architecture

### 4.1 Image Tokenization

```
IMAGES BECOME TOKENS in the LLM's sequence:

  Text sequence:     [BOS] "What"  "is"  "in"  "this"  "image"  "?"
  Visual tokens:     [IMG] v₁ v₂ v₃ ... v₅₇₆ [/IMG]
  Combined input:    [IMG] v₁...v₅₇₆ [/IMG] "What" "is" "in" ...

  The LLM treats visual tokens EXACTLY like text tokens:
  - Same attention mechanism
  - Visual tokens attend to text and vice versa
  - Causal mask applies normally (image before question)

TOKEN BUDGET:
  576 visual tokens from one image = 576 tokens of context used!
  For multi-image: 4 images × 576 = 2304 tokens for vision alone.
  
  HIGH-RES OPTIMIZATION:
  Dynamic resolution: resize to fit more patches only when needed.
  AnyRes (LLaVA-1.6): split image into tiles, encode each separately.
```

### 4.2 Interleaved Image-Text

```
MULTI-IMAGE / INTERLEAVED:

  "Here is photo 1: [IMG]v₁...v₅₇₆[/IMG]
   And here is photo 2: [IMG]v₅₇₇...v₁₁₅₂[/IMG]
   Which photo shows a cat?"

  The LLM can compare, contrast, and reason across multiple images.
  Each image is just another "paragraph" of visual tokens.

VIDEO (many frames):
  Sample K frames from video → K × 576 visual tokens.
  For K=8 (typical): 4608 visual tokens.
  With dynamic downsampling: reduce to ~1000 tokens for efficiency.
```

---

## 5. Summary

### 5.1 Formulas Quick Reference

**CLIP contrastive loss:**

```
L = -(1/N) Σᵢ log[exp(sim(Iᵢ,Tᵢ)/τ) / Σⱼ exp(sim(Iᵢ,Tⱼ)/τ)]
```

**ViT patch embedding:**

```
N_patches = (H/P) × (W/P)
z₀ = flatten(patches) × W_embed + pos_embed
```

**Projection:**

```
v_llm = MLP(v_vision) = GELU(v × W₁) × W₂     (d_vis → d_llm)
```

**Visual token count:**

```
N_visual = (H/P) × (W/P) = (336/14)² = 576 tokens per image
```

| Model | Vision Encoder | Projector | LLM | Visual Tokens |
|-------|---------------|-----------|-----|---------------|
| LLaVA-1.5 | CLIP ViT-L/14 | 2-layer MLP | Vicuna-7B | 576 |
| InternVL-2 | InternViT-6B | MLP | InternLM-7B | 256-1024 |
| GPT-4V | Custom | Integrated | GPT-4 | Unknown |
| Gemini | Custom | Early fusion | Gemini | Flexible |

### 5.2 Common Mistakes

```
❌ WRONG: The LLM "sees" the image directly
✓ RIGHT:  The LLM receives PROJECTED FEATURES from a vision encoder.
          It has no pixel-level access. It processes abstract visual tokens
          that have been transformed to be "language-like" by the projector.

❌ WRONG: Multimodal models need massive multimodal pretraining
✓ RIGHT:  LLaVA trains only Stage 2 (projector + LLM fine-tune) with
          665K examples and achieves strong results. The key insight:
          use a pretrained vision encoder + pretrained LLM + small bridge.

❌ WRONG: More visual tokens always helps
✓ RIGHT:  More tokens = more detail but also more context budget consumed.
          For simple images (solid color, single object): 64 tokens suffice.
          For complex scenes (dense text, charts): 2000+ tokens help.
          Dynamic resolution adapts token count to image complexity.

❌ WRONG: CLIP features capture everything about an image
✓ RIGHT:  CLIP is trained for image-TEXT matching (semantic similarity).
          It's weak at: spatial reasoning, counting, fine-grained details.
          This is why multimodal LLMs still struggle with "how many X?"
          and spatial relationship questions.
```

---

## 6. Exercises

1. **Token Budget**: An image is 672×672 with patch size P=14. How many visual tokens does it produce? If context window is 8192 and you have 2 such images + 500 text tokens + 500 response tokens, do they fit?

2. **Projection Parameters**: For d_vis=1024 and d_llm=4096 with a 2-layer MLP (hidden=4096): compute total projection parameters. Compare to an 8B LLM — what fraction is the projector?

3. **CLIP Loss**: For a batch of N=4 image-text pairs with similarities: sim(I₁,T₁)=0.9, sim(I₁,T₂)=0.3, sim(I₁,T₃)=0.1, sim(I₁,T₄)=0.2, τ=0.07: compute the contrastive loss for image I₁.

4. **Multi-Image**: You want to process a 10-page PDF document where each page is a 448×448 image. With P=14, how many total visual tokens? Can this fit in a 128K context window?
