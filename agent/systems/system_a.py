"""
System A: Single-Agent with Reflection

One agent identity handles EDA, feature analysis, rule generation, and self-reflection
in a single continuous agentic loop. Reflection is injected as a second user turn that
prompts the agent to critique and revise its initial rules.

Phases:
  Phase 1 — EDA + Analysis + Rule Generation (agentic loop with execute_python tool)
  Phase 2 — Reflection: critique and revise rules (continuation of same conversation)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.executor import execute_python
from utils.llm import TOOLS, run_agent_loop
from utils.rules import parse_rules

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_TASK_PROMPT = """\
Dataset path: {dataset_path}

Analyze this dataset and generate classification rules. \
Use execute_python to explore the data thoroughly before generating rules. \
End your response with the rules JSON in a ```json code block."""

_REFLECTION_PROMPT = """\
Review the rules you just generated.

Critique them on these points:
1. Is each rule well-supported by patterns you observed in the data?
2. Do the rules follow the contract (max 3 rules, max 2 conditions, valid operators)?
3. Are there predictive patterns you discovered but did NOT capture in a rule?

Then output your FINAL revised rules as JSON in a ```json code block. \
If the rules are already good, output them unchanged."""


def run(dataset_path: str, budget: int = 12) -> dict:
    """Run System A and return {"output": rules_list_or_None, "log": {...}}."""
    log: dict = {"system": "A", "llm_calls": [], "tool_calls": []}
    system_prompt = (_PROMPTS_DIR / "system_a.txt").read_text(encoding="utf-8")

    # Phase 1: EDA + Analysis + Rule Generation
    messages: list = [
        {"role": "user", "content": _TASK_PROMPT.format(dataset_path=dataset_path)}
    ]
    phase1_text, c1 = run_agent_loop(
        messages, system_prompt, TOOLS, max(1, budget - 3), log, execute_python
    )

    # Phase 2: Reflection
    messages.append({"role": "user", "content": _REFLECTION_PROMPT})
    final_text, c2 = run_agent_loop(
        messages, system_prompt, TOOLS, max(1, budget - c1), log, execute_python
    )

    # Parse rules from reflection output first, fall back to phase 1 output
    output = parse_rules(final_text) or parse_rules(phase1_text)
    log["total_calls"] = c1 + c2

    return {"output": output, "log": log}
