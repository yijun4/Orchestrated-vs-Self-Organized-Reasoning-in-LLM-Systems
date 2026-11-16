"""
System B: Multi-Agent, No Reflection

Two specialized agents in a strict sequential pipeline. No critic or revision step.

Agents:
  1. Analysis Agent  — EDA + feature analysis using execute_python
  2. Rule Gen Agent  — generates rules from the analysis text (no tools)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.executor import execute_python
from utils.llm import TOOLS, run_agent_loop
from utils.rules import parse_rules

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_ANALYSIS_TASK = "Dataset path: {dataset_path}\n\nAnalyze this dataset thoroughly and end with the ANALYSIS SUMMARY."

_RULES_TASK = """\
Here is the dataset analysis report:

{analysis}

Based on this analysis, generate classification rules and output them as JSON in a ```json code block."""


def run(dataset_path: str, budget: int = 12) -> dict:
    """Run System B and return {"output": rules_list_or_None, "log": {...}}."""
    log: dict = {"system": "B", "llm_calls": [], "tool_calls": []}

    analysis_prompt = (_PROMPTS_DIR / "system_b_analysis.txt").read_text(encoding="utf-8")
    rules_prompt = (_PROMPTS_DIR / "system_b_rules.txt").read_text(encoding="utf-8")

    # Agent 1: Analysis (uses execute_python tools)
    messages_1: list = [
        {"role": "user", "content": _ANALYSIS_TASK.format(dataset_path=dataset_path)}
    ]
    analysis_text, c1 = run_agent_loop(
        messages_1, analysis_prompt, TOOLS, budget // 2, log, execute_python
    )

    # Agent 2: Rule Generation (text-only, no tools)
    messages_2: list = [
        {"role": "user", "content": _RULES_TASK.format(analysis=analysis_text)}
    ]
    final_text, c2 = run_agent_loop(
        messages_2, rules_prompt, [], max(1, budget - c1), log, None
    )

    output = parse_rules(final_text)
    log["total_calls"] = c1 + c2

    return {"output": output, "log": log}
