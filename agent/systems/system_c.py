"""
System C: Multi-Agent with Reflection (Critic)

Same Analysis → Rule Gen pipeline as System B, plus a Critic Agent that reviews
the rules and triggers up to 2 revision cycles.

Agents:
  1. Analysis Agent  — EDA + feature analysis (identical to System B Agent 1)
  2. Rule Gen Agent  — generates initial rules from analysis text
  3. Critic Agent    — reviews rules; if verdict is REVISE, triggers Rule Gen Agent again
                       (max 2 critique-revision cycles)
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

_REVISION_TASK = """\
Here is the dataset analysis report:

{analysis}

Previous rules that need revision:
{rules}

Critic's feedback:
{critique}

Revise the rules based on the feedback and output the updated rules as JSON in a ```json code block."""

_CRITIC_TASK = """\
## Dataset Analysis
{analysis}

## Current Rules
{rules}"""


def run(dataset_path: str, budget: int = 12) -> dict:
    """Run System C and return {"output": rules_list_or_None, "log": {...}}."""
    log: dict = {"system": "C", "llm_calls": [], "tool_calls": [], "critique_cycles": []}

    analysis_prompt = (_PROMPTS_DIR / "system_b_analysis.txt").read_text(encoding="utf-8")
    rules_prompt = (_PROMPTS_DIR / "system_b_rules.txt").read_text(encoding="utf-8")
    critic_prompt = (_PROMPTS_DIR / "system_c_critic.txt").read_text(encoding="utf-8")

    calls_used = 0

    # Agent 1: Analysis
    messages_1: list = [
        {"role": "user", "content": _ANALYSIS_TASK.format(dataset_path=dataset_path)}
    ]
    analysis_text, c1 = run_agent_loop(
        messages_1, analysis_prompt, TOOLS, budget // 3, log, execute_python
    )
    calls_used += c1

    # Agent 2: Initial Rule Generation
    messages_2: list = [
        {"role": "user", "content": _RULES_TASK.format(analysis=analysis_text)}
    ]
    rules_text, c2 = run_agent_loop(
        messages_2, rules_prompt, [], min(3, budget - calls_used), log, None
    )
    calls_used += c2

    # Agent 3: Critic + Revision loop (max 2 cycles)
    for cycle in range(2):
        if calls_used >= budget - 1:
            break

        # Critique current rules
        messages_c: list = [
            {"role": "user", "content": _CRITIC_TASK.format(analysis=analysis_text, rules=rules_text)}
        ]
        critique_text, c3 = run_agent_loop(
            messages_c, critic_prompt, [], 1, log, None
        )
        calls_used += c3
        log["critique_cycles"].append({"cycle": cycle + 1, "critique": critique_text[:500]})

        if "ACCEPT" in critique_text:
            break

        if calls_used >= budget:
            break

        # Revise rules based on critique
        messages_r: list = [
            {
                "role": "user",
                "content": _REVISION_TASK.format(
                    analysis=analysis_text, rules=rules_text, critique=critique_text
                ),
            }
        ]
        rules_text, c4 = run_agent_loop(
            messages_r, rules_prompt, [], min(2, budget - calls_used), log, None
        )
        calls_used += c4

    output = parse_rules(rules_text)
    log["total_calls"] = calls_used

    return {"output": output, "log": log}
