# Understanding LLM Reasoning: Chain-of-Thought, Tree Search, and Test-Time Compute

*From simple prompting to o1-style deliberative reasoning — the mathematics of thinking before answering*

---

**LLM Reasoning** covers the emerging paradigm where models allocate additional compute at inference time to solve hard problems. This includes Chain-of-Thought (CoT), Tree-of-Thought (ToT), Monte Carlo Tree Search (MCTS) for LLMs, and the "process reward model" framework behind OpenAI o1 / o3.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 System 1 vs System 2](#11-system-1-vs-system-2)
   - [1.2 Test-Time Compute Scaling](#12-test-time-compute-scaling)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [Chain-of-Thought (CoT)](#2-chain-of-thought-cot)
   - [2.1 Why CoT Works](#21-why-cot-works)
   - [2.2 Formal Analysis](#22-formal-analysis)
3. [Tree-of-Thought (ToT)](#3-tree-of-thought-tot)
   - [3.1 Search Tree Structure](#31-search-tree-structure)
   - [3.2 BFS vs DFS Strategies](#32-bfs-vs-dfs-strategies)
4. [Monte Carlo Tree Search for LLMs](#4-monte-carlo-tree-search-for-llms)
   - [4.1 UCB1 Formula](#41-ucb1-formula)
   - [4.2 Selection-Expansion-Simulation-Backprop](#42-selection-expansion-simulation-backprop)
5. [Process Reward Models (PRM)](#5-process-reward-models-prm)
   - [5.1 Outcome vs Process Rewards](#51-outcome-vs-process-rewards)
   - [5.2 Step-Level Verification](#52-step-level-verification)
6. [Best-of-N Sampling](#6-best-of-n-sampling)
   - [6.1 Mathematics of Best-of-N](#61-mathematics-of-best-of-n)
   - [6.2 Scaling with N](#62-scaling-with-n)
7. [Summary](#7-summary)
   - [7.1 Formulas Quick Reference](#71-formulas-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 System 1 vs System 2

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1 (fast, standard LLM):                             │
├─────────────────────────────────────────────────────────────┤
│  Input: "What is 2+3?"                                      │
│  Output: "5"                                                 │
│  FLOPs: 2N (one forward pass)                               │
│  Time: ~10ms                                                 │
│  → Fine for simple questions!                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 2 (slow, deliberative reasoning):                   │
├─────────────────────────────────────────────────────────────┤
│  Input: "Prove that √2 is irrational"                       │
│  Process:                                                    │
│    Step 1: "Assume √2 = p/q is rational..."                 │
│    Step 2: "Then 2q² = p²..."                               │
│    Step 3: "So p² is even, meaning p is even..."            │
│    Step 4: "Let p = 2k, then 2q² = 4k²..."                 │
│    Step 5: "So q² = 2k², meaning q is even..."             │
│    Step 6: "Contradiction: p/q not in lowest terms. QED"    │
│                                                             │
│  FLOPs: 2N × T_reasoning (many forward passes)              │
│  Time: ~60 seconds                                           │
│  → Much more compute, but solves HARD problems!             │
│                                                             │
│  KEY INSIGHT: trade MORE COMPUTE at inference for           │
│  BETTER QUALITY on hard problems.                           │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: System 1 is answering "What's your name?" instantly. System 2 is doing a complex tax calculation — you need scratch paper, check your work, try different approaches.

### 1.2 Test-Time Compute Scaling

```
SCALING LAWS (two orthogonal axes):

  Traditional: scale TRAINING compute (bigger model, more data)
    L(C_train) = A / C_train^α   ← Chinchilla et al.
  
  New: scale INFERENCE compute (more thinking per query)
    Accuracy(C_infer) increases with C_infer  ← o1 paradigm
    For GSM8K math: from 60% (1-shot) → 95% (with search/verification)

  ┌─────────────────────────────────────────────────────────┐
  │  Accuracy                                               │
  │  100% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●  (with MCTS)  │
  │                                     ╱                   │
  │   90% ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●╱    (best-of-N=256)  │
  │                             ╱                           │
  │   80% ─ ─ ─ ─ ─ ─ ─ ─●╱          (best-of-N=16)      │
  │                    ╱                                    │
  │   70% ─ ─ ─ ─●╱                   (CoT)               │
  │           ╱                                             │
  │   60% ●╱                           (1-shot, no CoT)    │
  │        └─────────────────────────────────────────────── │
  │        1×    4×    16×   64×   256×  1024×              │
  │                  Test-time compute multiplier            │
  └─────────────────────────────────────────────────────────┘
```

### 1.3 Pipeline Summary

```
METHODS BY INCREASING SOPHISTICATION:

Level 1: CHAIN-OF-THOUGHT (CoT)
  One linear trace of reasoning steps
  Cost: T_reason × 2N FLOPs
  
Level 2: BEST-OF-N SAMPLING
  Generate N independent solutions, pick best
  Cost: N × T × 2N FLOPs + N × reward_model_cost
  
Level 3: TREE-OF-THOUGHT (ToT)
  Branch at decision points, evaluate branches
  Cost: varies (BFS/DFS exploration budget)
  
Level 4: MCTS + PROCESS REWARD MODEL (o1-style)
  Full search tree with per-step verification
  Cost: very high (hundreds of forward passes per problem)
```

---

## 2. Chain-of-Thought (CoT)

### 2.1 Why CoT Works

```
┌─────────────────────────────────────────────────────────────┐
│  WITHOUT CoT:                                               │
│  Input:  "If John has 3 apples and buys 2 more, then       │
│           gives half to Mary, how many does John have?"     │
│  Output: "3" ← WRONG (model skips intermediate steps)      │
│                                                             │
│  WITH CoT:                                                  │
│  Input:  Same question + "Let's think step by step."        │
│  Output: "John starts with 3 apples.                        │
│           He buys 2 more: 3+2 = 5.                          │
│           He gives half to Mary: 5/2 = 2.5                  │
│           Since we can't have half apples: John keeps 3,    │
│           gives 2 to Mary."                                  │
│  Answer: "3" ← CORRECT (with reasoning visible!)            │
└─────────────────────────────────────────────────────────────┘

WHY IT WORKS (mechanistic explanation):

  1. SERIAL COMPUTATION: Transformers compute in parallel over positions.
     Without CoT, the model has only ONE forward pass to solve the problem.
     With CoT, each reasoning step uses ALL model depth → more serial compute.

  2. WORKING MEMORY: intermediate results stored in generated tokens
     serve as "external memory" — the model can refer back to them.

  3. ERROR PROPAGATION: errors in early steps are visible in context,
     allowing the model to self-correct (though not guaranteed).
```

### 2.2 Formal Analysis

```
COMPOSITIONAL REASONING as function composition:

  Problem: f(x) = f_3(f_2(f_1(x)))   (3 steps needed)
  
  WITHOUT CoT: model must compute f(x) in ONE forward pass
    P_correct ≤ P(f₁ correct) × P(f₂ correct | f₁) × P(f₃ correct | f₂)
    Even if each step is 95% accurate:
    P_correct ≤ 0.95³ = 0.857  (bounded by depth)
  
  WITH CoT: model computes f₁, writes result, then computes f₂, etc.
    Each step has FULL model capacity for ONE sub-problem.
    P_correct ≈ 0.99 × 0.99 × 0.99 = 0.97 (much better!)
  
  FORMAL RESULT (Feng et al. 2023):
    Constant-depth Transformers CANNOT solve TC⁰-hard problems.
    CoT with T reasoning tokens can simulate T steps of computation.
    → CoT provably increases the computational class the model can solve.
```

---

## 3. Tree-of-Thought (ToT)

### 3.1 Search Tree Structure

```
┌─────────────────────────────────────────────────────────────┐
│  CoT = ONE path through reasoning space                     │
│  ToT = MULTIPLE paths explored, best selected              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    [Problem]                                 │
│                   ╱    │    ╲                                │
│              Step1a  Step1b  Step1c  ← generate candidates  │
│              ╱   ╲     │      ╲                             │
│         Step2a  Step2b Step2c  Step2d ← expand promising    │
│           │       │      │                                   │
│        [eval]  [eval]  [eval]   ← evaluate with LLM/PRM    │
│           │                                                  │
│         Step3a  ← continue best branch                      │
│           │                                                  │
│        [Answer]                                              │
│                                                             │
│  At each node: generate K candidates, evaluate, prune.     │
│  Keep only top-B branches alive (beam width).              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 BFS vs DFS Strategies

```
BFS (Breadth-First):
  Evaluate ALL step-1 candidates before going to step 2.
  + Doesn't miss globally-best partial solutions
  - Memory: O(K^depth) nodes alive
  
  Best for: short reasoning (depth ≤ 4-5), many valid approaches
  
DFS (Depth-First):
  Explore ONE branch fully, backtrack if it fails.
  + Memory: O(depth) nodes alive
  - Can get stuck in unpromising subtree
  
  Best for: deep reasoning (proof search), cheap evaluation

BEAM SEARCH (hybrid):
  Keep top-B partial solutions at each level.
  + Bounded memory: O(B × depth)
  + Good balance between exploration and exploitation
  - May still miss globally-best solution if B too small
  
  Most commonly used for LLM reasoning (B=5-10 typical).
```

---

## 4. Monte Carlo Tree Search for LLMs

### 4.1 UCB1 Formula

```
SELECTION: which node to expand next?

  UCB1(node) = Q̄(node) + c × √(ln N_parent / N_node)
               ──────────   ─────────────────────────────
               exploitation        exploration

  Q̄(node)   = average reward from simulations through this node
  N_parent  = number of visits to parent node
  N_node    = number of visits to this node
  c         = exploration constant (√2 for theory, tuned in practice)

INTUITION:
  ┌────────────────────────────────────────────────────────────┐
  │  First term (exploitation):                                │
  │    Prefer nodes with high average reward                   │
  │    "Go where we've seen good results"                      │
  │                                                            │
  │  Second term (exploration):                                │
  │    Prefer nodes rarely visited (high N_parent/N_node)      │
  │    "Try things we haven't explored much"                   │
  │                                                            │
  │  As N_node → ∞: exploration term → 0 (focus on best)      │
  │  As c → 0: pure exploitation (greedy)                      │
  │  As c → ∞: pure exploration (random)                       │
  └────────────────────────────────────────────────────────────┘
```

### 4.2 Selection-Expansion-Simulation-Backprop

```
MCTS for LLM reasoning (one iteration):

Step 1: SELECTION
  Start at root, traverse tree using UCB1 until reaching a leaf.
  At each node, pick child with highest UCB1 score.

Step 2: EXPANSION
  At the leaf, generate K possible next reasoning steps using the LLM.
  Each becomes a new child node.

Step 3: SIMULATION (rollout)
  From each new child, complete the reasoning trace using the LLM
  (possibly with cheaper/faster generation strategy).
  Get final answer.

Step 4: EVALUATION
  Score the final answer:
  - Process Reward Model (PRM): score each STEP
  - Outcome Reward Model (ORM): score final answer only
  - Self-verification: ask model "Is this correct?"

Step 5: BACKPROPAGATION
  Update Q̄ and N for all nodes along the path:
  For each node in path:
    N_node += 1
    Q̄_node = (Q̄_old × (N-1) + reward) / N

REPEAT for budget iterations (typically 100-1000).
RETURN: answer from most-visited root child.

COMPUTE COST:
  Iterations × (selection + expansion + rollout) FLOPs
  = I × (O(depth) + K×2N + T_rollout×2N)
  For I=200, K=5, T=500: ~100,000 × 2N FLOPs per problem
  ≈ 200× more compute than single-pass inference
```

---

## 5. Process Reward Models (PRM)

### 5.1 Outcome vs Process Rewards

```
┌─────────────────────────────────────────────────────────────┐
│  OUTCOME REWARD MODEL (ORM):                                │
│  Score the FINAL ANSWER only.                               │
│                                                             │
│  R_ORM(answer) ∈ [0, 1]                                    │
│  "Is the final answer correct?"                             │
│                                                             │
│  Problem: A wrong reasoning path can arrive at right answer │
│  by luck. ORM can't distinguish good vs lucky reasoning.    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PROCESS REWARD MODEL (PRM):    ← used in o1                │
│  Score EACH STEP of the reasoning.                          │
│                                                             │
│  R_PRM(step_1) = 0.95  "Correct first step"                │
│  R_PRM(step_2) = 0.88  "Correct second step"               │
│  R_PRM(step_3) = 0.12  "ERROR detected at step 3!"         │
│  R_PRM(step_4) = 0.60  "Uncertain (built on bad step 3)"   │
│                                                             │
│  ADVANTAGE: Can detect WHICH step went wrong.               │
│  → Prune bad branches early (don't waste compute)          │
│  → Provide credit assignment for training                   │
└─────────────────────────────────────────────────────────────┘

FORMAL DEFINITION:
  PRM: (problem, step_1, ..., step_t) → r_t ∈ [0, 1]
  
  Trained on human annotations:
    "Is this reasoning step correct given the previous steps?"
    Labels: {correct, incorrect, neutral}
  
  Training loss (binary cross-entropy per step):
  L_PRM = −Σₜ [yₜ log r̂ₜ + (1−yₜ) log(1−r̂ₜ)]
```

### 5.2 Step-Level Verification

```
USING PRM TO GUIDE SEARCH:

  Step 1: "Let x = 2y + 3"
  PRM score: 0.92 → CONTINUE (this step looks correct)

  Step 2 (Branch A): "Substituting, 2(2y+3) = 10"
  PRM score: 0.85 → CONTINUE

  Step 2 (Branch B): "Therefore y = x/2 − 3/2"
  PRM score: 0.91 → CONTINUE (valid but different approach)

  Step 3 (from A): "4y + 6 = 10, so y = 1"
  PRM score: 0.95 → HIGH CONFIDENCE → likely correct

  Step 3 (from B): "Plugging into equation 2: y = 7"  
  PRM score: 0.15 → ERROR DETECTED → PRUNE THIS BRANCH!

ADVANTAGES OF STEP-LEVEL VERIFICATION:
  1. Early pruning: bad steps detected → don't waste compute
  2. Credit assignment: know WHERE the error is
  3. Better search guidance: UCB1 uses PRM scores as Q-values
  4. Training signal: can train on process data (not just outcomes)
```

---

## 6. Best-of-N Sampling

### 6.1 Mathematics of Best-of-N

```
SIMPLEST REASONING STRATEGY:
  Generate N independent solutions, score each, return the best.

FORMAL SETUP:
  For each sample i ∈ {1,...,N}:
    y_i ~ π_θ(·|x)       (sample full response)
    r_i = R(x, y_i)      (score with reward model)
  
  Return: y* = y_{argmax_i r_i}   (best scoring response)

PROBABILITY OF CORRECT ANSWER:
  If P(correct for single sample) = p:
  P(at least one correct out of N) = 1 − (1−p)ᴺ

  EXAMPLE:
    p = 0.3 (30% per attempt)
    N = 1:   P(correct) = 0.30
    N = 5:   P(correct) = 1 − 0.7⁵ = 0.83
    N = 10:  P(correct) = 1 − 0.7¹⁰ = 0.97
    N = 50:  P(correct) = 1 − 0.7⁵⁰ = 0.99998

  But: this assumes a PERFECT reward model.
  With imperfect RM (accuracy q):
    P(select correct) ≈ 1 − (1−p×q)ᴺ   (approximately)
```

### 6.2 Scaling with N

```
┌─────────────────────────────────────────────────────────────┐
│  BEST-OF-N SCALING CURVE (GSM8K math benchmark):            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  N=1:     60% accuracy (baseline)                           │
│  N=4:     72% accuracy                                      │
│  N=16:    81% accuracy                                      │
│  N=64:    88% accuracy                                      │
│  N=256:   92% accuracy                                      │
│  N=1024:  94% accuracy  (diminishing returns)               │
│                                                             │
│  COST:  N × (generation_cost + RM_scoring_cost)             │
│                                                             │
│  TRADE-OFF:                                                 │
│  N=16 is often the sweet spot:                              │
│    2× compute → 20% accuracy improvement (great ROI)       │
│  N=1024:                                                    │
│    1000× compute → 34% improvement (diminishing returns)   │
└─────────────────────────────────────────────────────────────┘

WHEN TO USE BEST-OF-N vs MCTS:
  Best-of-N: simple, embarrassingly parallel, no tree structure
    Good for: moderate difficulty, abundant compute budget
  
  MCTS: complex, sequential, builds search tree
    Good for: very hard problems (math olympiad, formal proofs)
    Can explore 10× more efficiently by pruning bad branches
```

---

## 7. Summary

### 7.1 Formulas Quick Reference

**Chain-of-Thought compute:**

```
C_CoT = T_reasoning × 2N FLOPs (T=100-2000 reasoning tokens)
```

**Best-of-N probability:**

```
P(correct) = 1 − (1−p)ᴺ   (with perfect reward model)
```

**UCB1 (MCTS selection):**

```
UCB1 = Q̄ + c × √(ln N_parent / N_child)
```

**PRM loss:**

```
L_PRM = −Σₜ [yₜ log r̂ₜ + (1−yₜ) log(1−r̂ₜ)]   (per-step BCE)
```

| Method | Compute multiplier | Accuracy gain | Parallelisable |
|--------|-------------------|---------------|----------------|
| CoT (1 trace) | 2-5× | +10-15% | No |
| Best-of-N (N=16) | 16× | +20-25% | Yes |
| ToT (beam=5, depth=4) | ~100× | +25-30% | Partially |
| MCTS (200 iterations) | 200-500× | +30-40% | Partially |
| o1-style (full) | 1000×+ | +40-50% | No |

### 7.2 Common Mistakes

```
❌ WRONG: CoT works because the model "thinks" in the hidden states
✓ RIGHT:  CoT works because intermediate tokens provide WORKING MEMORY.
          The model can attend back to computed intermediate results.
          Without generating them, the computation is lost after one pass.

❌ WRONG: More reasoning tokens always helps
✓ RIGHT:  There's diminishing returns and even NEGATIVE effects at extreme
          lengths. The model can get confused by its own verbose reasoning.
          Optimal CoT length is problem-dependent.

❌ WRONG: Best-of-N with N=1000 is always better than MCTS with 1000 nodes
✓ RIGHT:  MCTS is more efficient because it PRUNES bad branches early.
          Best-of-N generates 1000 complete solutions (wasteful if most fail
          at step 2). MCTS explores 1000 NODES, each a partial solution.

❌ WRONG: The PRM needs to be a separate model from the policy
✓ RIGHT:  Self-verification (asking the same model "is this step correct?")
          works surprisingly well as a cheap PRM substitute.
          But dedicated PRMs trained on human labels are more reliable.
```

---

## 8. Exercises

1. **Best-of-N**: If base model accuracy is p=0.4 on MATH benchmark, compute P(at least one correct) for N=1, 4, 16, 64, 256. At what N do you exceed 99%?

2. **UCB1 Trade-off**: Two branches in MCTS: Node A has Q̄=0.7, N=20. Node B has Q̄=0.5, N=3. Parent has N=50. With c=√2, which node is selected? What if c=0.5?

3. **CoT Compute**: A model with N=8B params generates 500 reasoning tokens before answering. Compare total FLOPs to single-shot (answer in 10 tokens). How many "effective passes" does CoT use?

4. **PRM vs ORM**: Consider a 5-step math solution where step 3 is wrong but the final answer is accidentally correct. How does ORM score this? How does PRM score it? Which gives better training signal for the LLM?
