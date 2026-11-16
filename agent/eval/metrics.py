"""Evaluation metrics for generated rule sets."""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def compute_f1(rules: list, df: pd.DataFrame, label_col: str = "label") -> float:
    """Apply rules to df and compute binary F1 (high_risk as positive class)."""
    from utils.rules import apply_rules

    if not rules:
        return 0.0
    preds = apply_rules(rules, df)
    truth = df[label_col].tolist()
    return float(
        f1_score(truth, preds, pos_label="high_risk", average="binary", zero_division=0)
    )


def compute_complexity(rules: list) -> tuple[int, float]:
    """Return (num_rules, avg_conditions_per_rule)."""
    if not rules:
        return 0, 0.0
    n = len(rules)
    avg = sum(len(r.get("conditions", [])) for r in rules) / n
    return n, round(avg, 3)


def compute_faithfulness(rules: list, ground_truth_rules: list) -> float:
    """Jaccard similarity of (feature, op, threshold) triples between generated and GT rules.

    A score of 1.0 means the generated rules use exactly the same conditions as GT.
    A score of 0.0 means no overlap.
    """
    if not rules or not ground_truth_rules:
        return 0.0

    def to_triples(rule_list: list) -> set:
        triples = set()
        for r in rule_list:
            for c in r.get("conditions", []):
                feat = c.get("feature")
                op = c.get("op")
                thresh = c.get("threshold")
                if feat and op and thresh is not None:
                    triples.add((feat, op, float(thresh)))
        return triples

    gen = to_triples(rules)
    gt = to_triples(ground_truth_rules)
    union = len(gen | gt)
    if union == 0:
        return 0.0
    return round(len(gen & gt) / union, 4)


def compute_stability(f1_scores: list) -> float:
    """Standard deviation of F1 scores across repeated runs."""
    if len(f1_scores) < 2:
        return 0.0
    return round(float(np.std(f1_scores, ddof=1)), 4)
