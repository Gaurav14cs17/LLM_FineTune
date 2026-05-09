# Understanding LLM Agents and Tool Use

*From ReAct prompting to function calling — building autonomous AI systems*

---

**LLM Agents** are systems that can take actions in the real world: search the web, run code, query databases, call APIs. Instead of just generating text, agents REASON about what action to take, EXECUTE it, OBSERVE the result, and iterate until the task is complete.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.1 What is an Agent?](#11-what-is-an-agent)
   - [1.2 Agent vs Chatbot](#12-agent-vs-chatbot)
   - [1.3 Pipeline Summary](#13-pipeline-summary)
2. [ReAct Framework](#2-react-framework)
   - [2.1 Thought-Action-Observation Loop](#21-thought-action-observation-loop)
   - [2.2 Formal Definition](#22-formal-definition)
3. [Function Calling](#3-function-calling)
   - [3.1 Tool Definitions (JSON Schema)](#31-tool-definitions-json-schema)
   - [3.2 Structured Output Generation](#32-structured-output-generation)
4. [Planning Strategies](#4-planning-strategies)
   - [4.1 Plan-then-Execute](#41-plan-then-execute)
   - [4.2 Reflexion (Self-Correction)](#42-reflexion-self-correction)
5. [Multi-Agent Systems](#5-multi-agent-systems)
   - [5.1 Role Specialisation](#51-role-specialisation)
   - [5.2 Communication Patterns](#52-communication-patterns)
6. [Safety and Grounding](#6-safety-and-grounding)
   - [6.1 Action Verification](#61-action-verification)
   - [6.2 Sandboxing](#62-sandboxing)
7. [Summary](#7-summary)
   - [7.1 Patterns Quick Reference](#71-patterns-quick-reference)
   - [7.2 Common Mistakes](#72-common-mistakes)
8. [Exercises](#8-exercises)

---

## 1. Overview

### 1.1 What is an Agent?

```
┌─────────────────────────────────────────────────────────────┐
│  STANDARD LLM:                                              │
│  Input → Generate text → Output                             │
│  (no interaction with the world)                            │
├─────────────────────────────────────────────────────────────┤
│  LLM AGENT:                                                 │
│  Input → Think → Act → Observe → Think → Act → ... → Output│
│                    ↕                                         │
│           ┌──────────────────┐                              │
│           │  EXTERNAL TOOLS  │                              │
│           │  - Web search    │                              │
│           │  - Code executor │                              │
│           │  - Calculator    │                              │
│           │  - Database      │                              │
│           │  - File system   │                              │
│           │  - API calls     │                              │
│           └──────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

> **Real-World Analogy**: A chatbot is like a person who can only talk. An agent is like a person with a phone, computer, and notepad — they can LOOK things up, COMPUTE results, and TAKE ACTIONS in the world, not just talk about them.

### 1.2 Agent vs Chatbot

| Capability | Chatbot | Agent |
|-----------|---------|-------|
| Text generation | ✅ | ✅ |
| Tool use | ❌ | ✅ |
| Multi-step reasoning | Limited | ✅ |
| World interaction | ❌ | ✅ |
| Self-correction | ❌ | ✅ (via observation) |
| Autonomous task completion | ❌ | ✅ |

### 1.3 Pipeline Summary

```
AGENT EXECUTION LOOP:

  1. USER provides task
      ↓
  2. LLM generates THOUGHT (reasoning about what to do)
      ↓
  3. LLM selects ACTION (which tool, what arguments)
      ↓
  4. RUNTIME executes action, returns OBSERVATION
      ↓
  5. LLM receives observation, DECIDES:
     - Task complete? → return final answer
     - Need more info? → go to step 2
      ↓
  6. REPEAT until done or max_iterations reached

MAX ITERATIONS: safety limit (typically 10-20 steps)
  Prevents infinite loops when agent gets stuck.
```

---

## 2. ReAct Framework

### 2.1 Thought-Action-Observation Loop

```
┌─────────────────────────────────────────────────────────────┐
│  EXAMPLE: "What was the GDP of France in 2023?"             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  THOUGHT 1: I need to find France's GDP for 2023.           │
│  I should search for this information.                      │
│                                                             │
│  ACTION 1: web_search("France GDP 2023")                    │
│                                                             │
│  OBSERVATION 1: "France's GDP in 2023 was approximately     │
│  $3.05 trillion, making it the 7th largest economy..."      │
│                                                             │
│  THOUGHT 2: I found the answer. France's GDP in 2023        │
│  was approximately $3.05 trillion. I can provide this       │
│  to the user with the source.                               │
│                                                             │
│  ACTION 2: finish("France's GDP in 2023 was approximately   │
│  $3.05 trillion (7th largest economy globally).")           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Formal Definition

```
REACT AGENT (Yao et al. 2022):

  State at step t: sₜ = (task, h₁, h₂, ..., hₜ₋₁)
    where hᵢ = (thoughtᵢ, actionᵢ, observationᵢ)

  At each step, the LLM generates:
    thoughtₜ ~ LLM(sₜ)           ← natural language reasoning
    actionₜ  ~ LLM(sₜ, thoughtₜ) ← structured tool call
  
  Environment returns:
    observationₜ = Environment(actionₜ)   ← tool execution result

  Termination:
    actionₜ = "finish(answer)" → return answer
    t > max_steps → return "unable to complete"

ADVANTAGES OVER ACTION-ONLY (no thought):
  1. Thoughts provide CHAIN-OF-THOUGHT reasoning (better decisions)
  2. Thoughts are INTERPRETABLE (humans can debug the agent)
  3. Thoughts can PLAN ahead ("I need to do X before Y")
  4. Thoughts allow SELF-REFLECTION ("that didn't work, try Z")
```

---

## 3. Function Calling

### 3.1 Tool Definitions (JSON Schema)

```
TOOL DEFINITION FORMAT (OpenAI-style):

{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name (e.g., 'Paris')"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature units"
      }
    },
    "required": ["city"]
  }
}

THE LLM MUST:
  1. Understand WHEN to call this tool (from description)
  2. Generate CORRECT arguments (from schema)
  3. Produce VALID JSON (parseable by runtime)
```

### 3.2 Structured Output Generation

```
HOW DOES THE LLM PRODUCE STRUCTURED TOOL CALLS?

METHOD 1: Special tokens (GPT-4, Claude)
  Training includes tool-call examples with special formatting:
  
  <tool_call>
  {"name": "get_weather", "arguments": {"city": "Paris"}}
  </tool_call>
  
  The model learns to produce these during SFT/RLHF training.

METHOD 2: Constrained decoding (Outlines, LMQL)
  During generation, MASK invalid tokens at each step:
  
  After generating '{"name": "get_weather", "arguments": {"city": '
  → Only allow string-start tokens (quotes, letters)
  → Prevent generating invalid JSON
  
  FORMAL: at each step t, restrict vocabulary V to valid continuations:
    P(wₜ | w<t) → P(wₜ | w<t) × 1[wₜ ∈ valid(w<t)]
    then renormalise.

METHOD 3: Fine-tuned function-calling model
  Train specifically on (instruction, tool_definitions, tool_call) triples.
  Models: GPT-4, Claude, Gorilla, Hermes-2.
```

---

## 4. Planning Strategies

### 4.1 Plan-then-Execute

```
┌─────────────────────────────────────────────────────────────┐
│  PLAN-THEN-EXECUTE (two-phase):                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE 1 — PLANNING:                                        │
│  LLM generates a high-level plan BEFORE taking any action:  │
│                                                             │
│  Task: "Book a flight from NYC to Paris for next Tuesday"   │
│  Plan:                                                       │
│    1. Search for available flights NYC→Paris on [date]      │
│    2. Filter by price and duration                          │
│    3. Select best option                                    │
│    4. Fill in passenger details                             │
│    5. Confirm booking                                       │
│                                                             │
│  PHASE 2 — EXECUTION:                                       │
│  Execute each step, adapting if needed:                      │
│    Step 1: call flight_search(...)                           │
│    Step 2: [observe results, filter]                         │
│    Step 3: [select — but what if no good options?]          │
│    → REPLAN: "No direct flights. Search with 1 layover."    │
│                                                             │
│  ADVANTAGE: reduces wasted actions (thinks before acting)   │
│  DISADVANTAGE: plan may not survive contact with reality    │
│  → Need ADAPTIVE replanning when observations surprise      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Reflexion (Self-Correction)

```
REFLEXION (Shinn et al. 2023):
  After a failed attempt, the agent REFLECTS on what went wrong
  and uses that reflection to improve the next attempt.

LOOP:
  Attempt 1: Execute task → evaluate → FAIL
      ↓
  Reflect: "I failed because I searched with wrong keywords.
            Next time, I should use more specific terms."
      ↓
  Attempt 2: Execute task (with reflection in context) → SUCCESS

FORMAL:
  Memory M = [reflection₁, reflection₂, ...]
  
  At attempt k:
    context = task + M + observation_history
    action ~ LLM(context)
  
  After failure:
    reflectionₖ = LLM("What went wrong? How to improve?", 
                       attempt_k_history)
    M.append(reflectionₖ)

EFFECTIVENESS:
  HumanEval (code): 67% → 91% with 3 Reflexion iterations
  AlfWorld (tasks): 54% → 97% with 5 iterations
```

---

## 5. Multi-Agent Systems

### 5.1 Role Specialisation

```
MULTI-AGENT ARCHITECTURE:

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ PLANNER  │───►│ EXECUTOR │───►│ REVIEWER │
  │ (GPT-4)  │    │(Code LLM)│    │(Critic)  │
  └──────────┘    └──────────┘    └──────────┘
       ↑                                │
       └────────── feedback ────────────┘

  PLANNER:  Breaks task into sub-tasks, creates plan
  EXECUTOR: Writes code, calls tools, takes actions
  REVIEWER: Checks output quality, provides feedback
  
  Each agent can be a DIFFERENT model optimised for its role:
    Planner: large reasoning model (GPT-4, o1)
    Executor: code-specialised model (Codex, DeepSeek-Coder)
    Reviewer: small fast model (GPT-3.5) for quick checks
```

### 5.2 Communication Patterns

```
DEBATE (multiple agents discuss):
  Agent A: proposes answer
  Agent B: critiques / proposes alternative
  Agent C: synthesises best parts of both
  → Higher quality than single agent (diverse perspectives)

DELEGATION (hierarchical):
  Manager agent receives complex task
  → Decomposes into sub-tasks
  → Assigns each to specialised worker agent
  → Collects results and synthesises final answer

COLLABORATIVE (shared workspace):
  Multiple agents work on different parts simultaneously
  Shared state: document, codebase, or database
  Each agent reads current state, contributes its part
  → Parallelism for complex tasks (writing, research)
```

---

## 6. Safety and Grounding

### 6.1 Action Verification

```
BEFORE executing any action, verify:

  1. PERMISSION CHECK:
     - Does this action require user confirmation?
     - Is it reversible? (read vs write vs delete)
     - Cost implications? (API calls with charges)
  
  2. ARGUMENT VALIDATION:
     - Do the arguments match the expected schema?
     - Are there injection attacks in string arguments?
     - Are numeric values within reasonable bounds?
  
  3. RATE LIMITING:
     - How many actions has the agent taken this session?
     - Is it in an infinite loop?
     - Has it exceeded the time/cost budget?

SAFETY LEVELS:
  Level 0: Read-only actions (search, lookup) → auto-execute
  Level 1: Reversible writes (create file, send draft) → auto-execute
  Level 2: Irreversible writes (send email, delete) → ASK USER
  Level 3: High-stakes (financial, access control) → ALWAYS ASK
```

### 6.2 Sandboxing

```
CODE EXECUTION SAFETY:
  Agents that run code MUST be sandboxed:

  ┌────────────────────────────────────────────────────────────┐
  │  SANDBOXING LAYERS:                                        │
  │                                                            │
  │  1. Container isolation (Docker, Firecracker):             │
  │     - No network access (unless explicitly allowed)        │
  │     - No filesystem access outside working directory       │
  │     - Resource limits: CPU, memory, disk, time             │
  │                                                            │
  │  2. Language-level restrictions:                            │
  │     - No imports of dangerous modules (os, subprocess)     │
  │     - No file I/O outside sandbox                         │
  │     - Timeout: kill after T seconds                        │
  │                                                            │
  │  3. Output validation:                                     │
  │     - Check for sensitive data in output                   │
  │     - Limit output size (prevent memory exhaustion)        │
  │     - Sanitise before showing to user                      │
  └────────────────────────────────────────────────────────────┘
```

---

## 7. Summary

### 7.1 Patterns Quick Reference

**ReAct loop:**

```
While not done and step < max_steps:
  thought = LLM(state)
  action = LLM(state + thought)
  observation = execute(action)
  state.append(thought, action, observation)
```

**Function call format:**

```
{"name": "tool_name", "arguments": {"param1": value1, "param2": value2}}
```

**Reflexion:**

```
For attempt in 1..K:
  result = execute(task, memory=reflections)
  if success: return result
  reflection = LLM("What went wrong?", attempt_history)
  reflections.append(reflection)
```

| Strategy | When to use | Compute cost |
|----------|-------------|-------------|
| Single ReAct | Simple tasks (1-5 steps) | Low |
| Plan-then-Execute | Complex multi-step tasks | Medium |
| Reflexion | Tasks that benefit from retry | Medium-High |
| Multi-Agent | Very complex, multi-domain | High |
| MCTS + Agent | Hard reasoning + actions | Very High |

### 7.2 Common Mistakes

```
❌ WRONG: The agent should always take actions — thinking is wasted tokens
✓ RIGHT:  THOUGHTS are critical. Without them, the agent makes random actions.
          The thought step is where the model PLANS and REASONS about what to do.
          Removing thoughts typically reduces success rate by 20-40%.

❌ WRONG: Give the agent access to all tools at once
✓ RIGHT:  Too many tools confuse the model (decision paralysis).
          Limit to 5-10 most relevant tools per task.
          Use a "tool router" to select relevant tools based on the query.

❌ WRONG: The agent can reliably call tools with complex nested arguments
✓ RIGHT:  Function calling is not 100% reliable, especially for:
          - Deeply nested JSON
          - Ambiguous parameter types
          - Tools with >5 parameters
          Use simple, flat tool schemas when possible.

❌ WRONG: Multi-agent systems are always better than single agents
✓ RIGHT:  Multi-agent adds communication overhead and coordination complexity.
          For simple tasks, a single well-prompted agent is more reliable.
          Multi-agent shines for tasks that genuinely require diverse expertise
          or parallelism (large codebases, research synthesis).
```

---

## 8. Exercises

1. **ReAct Trace**: Design a ReAct trace (thought, action, observation) for the task: "What is the stock price of Apple AND Microsoft today, and which is higher?" Include all steps with realistic observations.

2. **Tool Schema Design**: Design a JSON schema for a tool `execute_sql` that takes a database name, a SQL query, and an optional row limit. Include proper types and descriptions.

3. **Reflexion**: An agent tries to book a hotel but fails because it searched for "hotel in NYC" which returned too many results. Write a reflection that would help the next attempt succeed.

4. **Safety Classification**: Classify these actions by safety level (0-3): (a) web search, (b) create a file, (c) send an email to a colleague, (d) delete a database table, (e) charge a credit card. Justify each.
