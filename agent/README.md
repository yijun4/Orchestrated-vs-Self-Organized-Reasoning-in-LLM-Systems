# Orchestrated vs Self-Organized Reasoning in LLM Systems

A course project comparing two reasoning architectures in LLM-based systems on a rule-based credit risk classification task.

## What This Project Does

Modern LLM applications are built in two broad ways: a **single agent** that plans and self-corrects in one continuous loop, or a **multi-agent pipeline** where specialized agents hand off work to each other. This project asks: under a fixed compute budget and identical tools, does one architecture produce better and more consistent results?

Three systems are implemented and compared:

| System | Architecture | Reflection |
|--------|-------------|------------|
| A | Single-Agent | Yes (self-critique turn) |
| B | Multi-Agent (Analysis → Rule Gen) | No |
| C | Multi-Agent (Analysis → Rule Gen → Critic) | Yes (up to 2 revision cycles) |

Each system receives the same tabular credit-risk dataset and must output a set of classification rules (max 3 rules, max 2 conditions each). The same model, tools, and output schema are used across all systems. Evaluation runs each system 5 times to measure both quality and stability.

**Research questions:**
1. Does explicit agent decomposition improve rule quality over a single-agent loop? *(A vs C — architecture, both with reflection)*
2. Does reflection improve multi-agent output quality and consistency? *(B vs C — reflection, both multi-agent)*
3. What are the cost and stability trade-offs between architectures?

**Metrics:** F1 score, rule complexity, faithfulness to ground truth, and stability (F1 std across runs).

---

## Project Structure

```
agent/
├── data/
│   ├── generate.py          # synthetic dataset generation script
│   ├── train.csv            # generated at init (400 rows)
│   ├── test.csv             # generated at init (100 rows)
│   └── ground_truth.json    # 3 ground-truth rules used to create labels
├── utils/
│   ├── llm.py               # Anthropic API wrapper + agentic loop
│   ├── executor.py          # sandboxed Python code execution
│   └── rules.py             # rule parsing and application
├── prompts/
│   ├── system_a.txt         # single-agent system prompt
│   ├── system_b_analysis.txt
│   ├── system_b_rules.txt
│   └── system_c_critic.txt
├── systems/
│   ├── system_a.py          # single-agent + reflection
│   ├── system_b.py          # 2-agent pipeline, no reflection
│   └── system_c.py          # 2-agent + critic loop
├── eval/
│   ├── metrics.py           # F1, complexity, faithfulness, stability
│   └── run_eval.py          # evaluation orchestrator
├── logs/                    # per-run JSON traces (created at runtime)
├── results/                 # results.csv (created at runtime)
└── requirements.txt
```

---

## Setup

### 1. Python version

Python 3.10 or higher is required.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Generate the dataset

```bash
python data/generate.py
```

This creates `data/train.csv`, `data/test.csv`, and `data/ground_truth.json`. Output should look like:

```
Train: 400 rows  |  Test: 100 rows
Train label distribution:
  low_risk     289
  high_risk    111
Train missing in employment_years: 39
```

---

## Running the Evaluation

### Quick test (1 run, single system, ~5 API calls)

```bash
python eval/run_eval.py --runs 1 --systems B
```

### Full evaluation (5 runs × 3 systems, default settings)

```bash
python eval/run_eval.py
```

### Options

```
--runs      Number of runs per system  (default: 5)
--budget    Max LLM API calls per run  (default: 12)
--systems   Comma-separated systems    (default: A,B,C)
--model     Override the LLM model     (default: claude-haiku-4-5-20251001)
```

### Using a stronger model for final evaluation

```bash
python eval/run_eval.py --model claude-sonnet-4-6
```

### Example output

```
==================================================
System A  (5 runs, budget=12)
==================================================
  Run 1/5 ... F1=0.872  rules=3  faith=0.60  calls=8  (14.3s)
  Run 2/5 ... F1=0.891  rules=2  faith=0.40  calls=7  (12.1s)
  ...
  → Stability (F1 std across 5 runs): 0.0183

==================================================
SUMMARY (mean ± std across runs)
==================================================
        f1_mean  f1_std  stability  avg_rules  avg_faithfulness  avg_llm_calls
system
A         0.871   0.018      0.018        2.6             0.480          7.8
B         0.843   0.031      0.031        2.2             0.360          4.2
C         0.882   0.012      0.012        2.8             0.520          9.1
```

Results are saved to `results/results.csv`. Per-run logs (all LLM calls, tool calls, generated rules) are saved to `logs/system_X_runNN.json`.

---

## Model and Cost Notes

The default model is `claude-haiku-4-5-20251001` (cheapest, fastest). A full 5-run evaluation across all 3 systems costs approximately **$0.05–0.15** with Haiku and **$0.50–1.50** with Sonnet, depending on how much code the agents generate.

Use Haiku during development and switch to Sonnet for the final evaluation run you report in your write-up.
