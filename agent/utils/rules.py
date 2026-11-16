"""Rule parsing and application utilities."""
import json
import re
from typing import Optional

import pandas as pd

_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def parse_rules(text: str) -> Optional[list]:
    """Extract a rules list from LLM response text.

    Tries JSON code blocks first, then scans for any JSON object containing "rules".
    Returns the list of rule dicts, or None if parsing fails.
    """
    if not text:
        return None

    # 1. Try fenced code blocks: ```json ... ``` or ``` ... ```
    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pattern, text)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, dict) and "rules" in data:
                    return data["rules"]
            except (json.JSONDecodeError, ValueError):
                pass

    # 2. Scan for balanced { ... } blocks containing "rules"
    brace_count = 0
    start = -1
    last_valid = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_count == 0:
                start = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and "rules" in data:
                        last_valid = data["rules"]
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1

    return last_valid


def apply_rules(rules: list, df: pd.DataFrame) -> list:
    """Apply rule set to a DataFrame. First-match wins; default label is 'low_risk'.

    Each rule: {"conditions": [{"feature": str, "op": str, "threshold": float}], "label": str}
    """
    predictions = []
    for _, row in df.iterrows():
        label = "low_risk"
        for rule in rules:
            conditions = rule.get("conditions", [])
            if not conditions:
                continue
            match = True
            for cond in conditions:
                feat = cond.get("feature")
                op = cond.get("op")
                thresh = cond.get("threshold")
                if feat not in row.index or pd.isna(row[feat]):
                    match = False
                    break
                op_fn = _OPS.get(op)
                if op_fn is None or not op_fn(row[feat], thresh):
                    match = False
                    break
            if match:
                label = rule.get("label", "low_risk")
                break
        predictions.append(label)
    return predictions
