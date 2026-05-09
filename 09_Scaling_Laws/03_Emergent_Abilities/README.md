# 03 — Emergent Abilities and Phase Transitions

## 1. Formal Definition of Emergence

```
DEFINITION (Wei et al. 2022):
  An ability is EMERGENT if it is not present in smaller models but is present in larger models.
  Formally: there exists a threshold N* such that:
    performance(N) ≈ random_baseline    for N < N*
    performance(N) >> random_baseline  for N ≥ N*

MATHEMATICAL CHARACTERISATION:
  Let P(N) = performance on a task as a function of parameters.
  
  NON-EMERGENT (smooth scaling):
    P(N) = a × N^b  for some a, b > 0
    ∂P/∂N > 0 for all N → continuous improvement
  
  EMERGENT (sharp transition):
    P(N) ≈ P_chance     for N < N*
    P(N) >> P_chance    for N ≥ N*
    
    The derivative ∂P/∂N is VERY SMALL for N < N* and VERY LARGE near N*.

VISUAL:
  Performance ↑
              │                      ████████████████
              │               ███████
              │         ██████
              │     █████
  P_chance ───├─████
              └──────────────────────────────────────── log N →
                        ↑ threshold N* 

  Before N*: performance = random chance (task completely unsolved)
  After N*:  rapid improvement as model size increases
```

## 2. Phase Transition Mathematics — Percolation Theory Analogy

```
ANALOGY: emergence in LLMs is similar to phase transitions in physics.

PERCOLATION TRANSITION (bond percolation):
  Each bond in a lattice is occupied with probability p.
  At threshold p_c ≈ 0.5, a connected path suddenly appears (sharp transition).
  
  Below p_c: isolated clusters, no long-range connectivity
  Above p_c: giant connected component spanning the lattice

LLM CAPABILITY ANALOGY:
  Each additional parameter (or token during training) = adding one bond.
  "Connectivity" = ability to chain together sub-skills.
  
  MATHEMATICAL MODEL (toy):
    Suppose a complex task requires k sub-skills S₁,...,Sₖ.
    Model of size N solves each sub-skill with probability f(N).
    Task is solved when ALL sub-skills are solved:
    
    P_task(N) = ∏ᵢ f_i(N)
    
    If each f_i grows slowly (near 0 for small N):
    P_task ≈ 0  (product of small numbers)
    
    When N crosses the threshold so that each f_i ≈ 1:
    P_task → ∏ 1 = 1  (sudden jump!)
    
    The transition APPEARS SHARP but is actually a composition of smooth curves.

BIG BENCH HARD TASKS:
  Tasks like "chain of thought arithmetic" require multiple capabilities:
  1. Parse numbers correctly
  2. Apply operator
  3. Carry results across steps
  4. Format output correctly
  5. Stay within token budget
  
  P_task = P(parse) × P(operator) × P(carry) × P(format) × P(budget)
  
  If each sub-skill develops gradually from N=5B to N=50B:
    At 5B:  each P ≈ 0.5 → P_task ≈ 0.5^5 = 0.03 (3% ≈ random)
    At 50B: each P ≈ 0.9 → P_task ≈ 0.9^5 = 0.59 (59%, clearly above chance)
  
  The SHARP-LOOKING jump at ~50B is actually a product of 5 smooth curves.
```

## 3. Are Emergent Abilities Real? — Quantitative Analysis

```
SCHAEFFER ET AL. (2023): "Are Emergent Abilities of Large Language Models a Mirage?"

ARGUMENT: Emergence is an artifact of METRIC CHOICE, not model capability.

DICHOTOMOUS METRICS create artificial emergence:
  "Exact match accuracy" (0 or 1 per example):
    If task needs 5 correct answers to pass:
    With 4 correct: accuracy = 0 (fail)
    With 5 correct: accuracy = 1 (pass)
    
    Underlying capability (fraction correct per question) is SMOOTH.
    But binary pass/fail metric → sharp apparent threshold.

PROOF:
  Let Xᵢ ~ Bernoulli(p) for i=1,...,k (probability p of getting each sub-question right)
  Task "passes" if ∑ Xᵢ = k (all must be correct).
  
  P(pass | p) = p^k
  
  This function in p:
    For p < 0.9: P(pass) < 0.9^k ≈ 0 (for k=5: P < 0.59)
    At p = 0.95: P(pass) = 0.95^5 = 0.77
    At p = 0.99: P(pass) = 0.95 ≈ 1
  
  For small k: smooth. For large k: appears like a phase transition.

CONTINUOUS METRICS DON'T SHOW EMERGENCE:
  Using log-probability of correct answer (soft metric):
    P(correct answer) grows SMOOTHLY with N (no sharp transition)
  
  Using BLEU score (continuous):
    Grows smoothly with N (no sharp transition)
  
  CONCLUSION: Emergent abilities are a metric artifact, not a model capability artifact.
  The underlying capability IS SMOOTH, but the evaluation metric is discontinuous.
```

## 4. Predictability of Emergent Abilities

```
IF emergence is a metric artifact, can we PREDICT when it will occur?

METHOD 1 — From individual sub-task curves:
  Identify sub-tasks T₁,...,Tₖ that collectively constitute the complex task.
  Fit smooth curves P(Tᵢ | N) for each sub-task.
  Predict complex task performance: P_task(N) = f(P(T₁),...,P(Tₖ))
  
  EXAMPLE: 5-digit arithmetic
    Sub-tasks: (1-digit add, 2-digit add, 3-digit add, 4-digit add, 5-digit add)
    Each has a smooth sigmoid-like curve in N.
    5-digit curve = product of individual improvements → apparent sharp threshold.

METHOD 2 — Extrapolating power laws:
  Even below the apparent threshold:
    performance(N) = ε(N) ≈ c × N^d   for some small c, d
    (Model gets SOME fraction right — just below human-detectable threshold)
  
  Fit this curve to small-scale training runs.
  Extrapolate to predict when performance crosses a quality threshold.
  
  This is used in SCALING EXPERIMENTS:
    Train small models (N=100M, 1B, 10B)
    Fit power law
    Predict performance at N=100B WITHOUT training it!
    
    Accuracy of prediction: often within 5-10% for well-studied tasks.
    For rare "emergent" tasks: prediction is harder (exponents change).

METHOD 3 — Chain of Thought unlocking:
  Some tasks appear emergent at N=100B but are PRESENT at N=10B with Chain-of-Thought.
  
  CoT provides the model with the step-by-step "sub-task structure".
  This bypasses the need for the model to have internalised all sub-skills.
  → "Emergence" can be made to appear earlier with appropriate prompting!
```

## 5. Emergent Abilities in Practice — Examples and Thresholds

```
CONFIRMED EMERGENT ABILITIES (Wei et al. 2022):

Task                                │ Threshold N  │ What changes?
────────────────────────────────────┼──────────────┼────────────────────────────────
Multi-digit arithmetic (5-shot)     │ ~100B        │ Correct carryovers at all digits
3-digit ×3-digit multiplication     │ ~500B        │ Generalises to arbitrary numbers
Physics Q (college-level)           │ ~50B         │ Uses formulas correctly
Multi-step word problems (GSM8K)    │ ~50B         │ Chains multiple reasoning steps
Analogical reasoning (B-Big-Bench)  │ ~10B         │ Identifies abstract patterns
Word unscrambling                   │ ~100B        │ Orthographic manipulation

MATHEMATICAL THRESHOLD ANALYSIS for multi-digit arithmetic:
  5-digit addition requires:
    - Correct digit parsing (appears at ~1B params)
    - Correct carry logic (appears at ~10B params)
    - Consistent multi-step application (appears at ~100B params)
  
  Each step adds approximately one order of magnitude to the threshold.
  → k-step tasks need models ~10^k times larger than 1-step tasks (rough rule).

CHAIN-OF-THOUGHT EFFECTS ON THRESHOLDS:
  Task                        │ Without CoT  │ With CoT
  ────────────────────────────┼──────────────┼───────────
  Multi-step math (5 steps)   │ ~100B        │ ~20B  (5× smaller threshold!)
  Common sense reasoning      │ ~50B         │ ~7B
  Physics problems            │ ~100B        │ ~40B
  
  CoT provides the "sub-task structure" to the model explicitly,
  bypassing the need for the model to discover it internally.
  → Each explicit CoT step "saves" roughly one order of magnitude of model capacity.
```

---

## Exercises

1. For a task requiring k=3 sub-skills each with P(sub-skill) = N^{0.1} / N_max^{0.1}, compute the task success probability at N = 0.1, 0.5, and 1.0 × N_max. Show that the combined curve looks like a phase transition.
2. Explain the Schaeffer et al. argument mathematically: show that a task with "exact match" metric (pass if all k correct) exhibits apparent emergence even when individual probabilities grow smoothly.
3. If a 5-step task is "emergent" at N=100B, and each step individually is mastered at N=10B, 15B, 20B, 30B, 40B respectively: model P_task(N) as the product of 5 sigmoid functions each centered at the individual thresholds. At what N does P_task exceed 0.5?
