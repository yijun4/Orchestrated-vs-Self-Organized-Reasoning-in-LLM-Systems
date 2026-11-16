# Project Plan (Refined)

## Title

**Orchestrated vs Self-Organized Reasoning in LLM Systems for Rule-Based Risk Classification**

---

## 1. Research Objective

This project investigates:

> Under the same model capability and compute budget, does **externally structured reasoning (multi-agent systems)** outperform **self-organized reasoning (single-agent systems, e.g., Claude Code-style)** in terms of reliability and rule quality?

---

## 2. Research Questions

1. Does multi-agent (explicit structure) improve:

   - Rule quality?
   - Stability?
   - Interpretability?
2. What is the effect of **self-reflection**:

   - Does it improve performance?
   - Does it reduce or amplify differences between architectures?
3. With modern strong LLMs:

   - Is multi-agent decomposition still necessary?

---

## 3. System Design

We compare three system architectures, with reflection as a controlled variable.

---

### System A: Strong Single-Agent (Modern Baseline)

Simulates systems like Claude Code or Deep Research.

**Workflow:**
Task Input
↓
Planner (generate steps)
↓
EDA code generation → execution
↓
Reflection
↓
Feature analysis → code → execution
↓
Reflection
↓
Rule generation
↓
Final output

**Characteristics:**

- Single agent identity
- Iterative planning and execution
- Tool usage (Python execution)
- Optional self-reflection

---

### System B: Multi-Agent (Structured, No Reflection)

**Workflow:**
Data
↓
EDA Agent
↓
Feature Engineering Agent
↓
Rule Generation Agent
↓
Output

**Characteristics:**

- Explicit role decomposition
- Sequential pipeline
- No reflection (baseline)

---

### System C: Multi-Agent + Reflection

**Workflow:**
EDA → Feature → Rule
↓
Critic Agent
↓
Feedback → Regeneration (up to N iterations)

**Characteristics:**

- Structured decomposition
- Iterative refinement via critic
- Explicit error correction

---

## 4. Experimental Conditions

We evaluate 4 configurations:

| System Type  | Reflection |
| ------------ | ---------- |
| Single-Agent | ❌         |
| Single-Agent | ✅         |
| Multi-Agent  | ❌         |
| Multi-Agent  | ✅         |

---

## 5. Experimental Controls (Critical)

To ensure fair comparison:

### 1. Same Base Model

- Use identical LLM across all conditions

### 2. Compute Budget

- Same maximum number of LLM calls (e.g., 5)
- Or same token budget (e.g., 10k tokens)

### 3. Tool Access

All systems can:

- Generate Python code
- Execute code
- Observe outputs

### 4. Output Format (Standardized)

```json
{
  "rules": [...],
  "features": [...],
  "metrics": {...}
}

```


## 6. Task Design (Enhanced)

The task is  **rule-based risk classification on tabular data** , with increased complexity.

---

### Dataset Requirements (at least 3 of the following)

#### Feature Interactions

* Nonlinear or conditional relationships
* Example:
* `(debt_ratio > 0.6 AND income < 40000) → high risk`

#### Derived Features

* Example:
* `debt_to_income = debt / income`

#### Noisy Features

* Include irrelevant variables:
* `random_feature_1`
* `irrelevant_score`

#### Data Imperfections

* Missing values
* Skewed distributions

#### Rule Constraints (Important)

* Max 3 rules
* Max 2 conditions per rule

---

## 7. Evaluation Metrics

---

### 1. Accuracy

* Classification accuracy / F1 score

---

### 2. Rule Complexity

* Number of rules
* Average number of conditions per rule

---

### 3. Stability (Key Metric)

Run each configuration multiple times:

* Do generated rules vary?
* Performance variance?

---

### 4. Faithfulness

Do rules reflect actual data patterns?

Method:

* Apply generated rules to dataset
* Compare predictions with ground truth

---

### 5. Debuggability

Can we identify where failures occur?

* EDA stage
* Feature engineering stage
* Rule generation stage

---

### 6. Cost / Efficiency

* Token usage
* Latency

---

## 8. Logging & Trace Analysis

Record:

* Intermediate outputs
* Agent decisions
* Reflection steps

Analyze:

* Where errors originate
* Whether multi-agent improves traceability

---

## 9. Expected Outcomes

We consider three possible outcomes:

---

### Case 1: Multi-Agent Outperforms

* Especially under complex constraints or noisy data

---

### Case 2: Performance Gap Shrinks

* Strong LLMs can internally handle decomposition

---

### Case 3: Reflection Dominates

* Iterative refinement matters more than architecture

---

## 10. Contribution

This project contributes:

1. A  **controlled comparison framework** :

* Same compute budget
* Same tools
* Same model

2. A study of:

* Structured vs self-organized reasoning

3. Insights into:

* When multi-agent systems are beneficial in modern LLM settings

---

## 11. One-Sentence Summary

> This project compares externally orchestrated reasoning vs self-organized reasoning in LLM systems under constrained compute, focusing on reliability, stability, and interpretability.

---

## 12. (Optional) Ablation Study

To strengthen the study:

* Remove Feature Agent → observe impact
* Remove Reflection → observe impact

---

## 13. Implementation Notes (Optional)

* Use Python + tool execution environment
* Optional frameworks:
* LangGraph
* OpenAI function calling
* Claude tool-use API
