"""Generate synthetic credit-risk tabular dataset with known ground-truth rules.

Ground-truth rules (first-match wins, default = low_risk):
  Rule 1: debt_ratio > 0.6 AND income < 40000  → high_risk
  Rule 2: loan_amount > 15000 AND employment_years < 2 → high_risk
  Rule 3: credit_score < 580                   → high_risk

Features:
  income              — annual income in USD
  debt                — total outstanding debt in USD
  debt_ratio          — debt / income  (pre-computed derived feature)
  loan_amount         — requested loan amount in USD
  employment_years    — years at current employer  (~10% missing in train split)
  credit_score        — integer FICO-like score [300, 850]
  age                 — applicant age [18, 70]
  random_feature_1    — irrelevant noise (uniform)
  irrelevant_score    — irrelevant noise (normal)

Usage:
  python data/generate.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUND_TRUTH_RULES = [
    {
        "conditions": [
            {"feature": "debt_ratio", "op": ">", "threshold": 0.6},
            {"feature": "income", "op": "<", "threshold": 40000},
        ],
        "label": "high_risk",
    },
    {
        "conditions": [
            {"feature": "loan_amount", "op": ">", "threshold": 15000},
            {"feature": "employment_years", "op": "<", "threshold": 2},
        ],
        "label": "high_risk",
    },
    {
        "conditions": [
            {"feature": "credit_score", "op": "<", "threshold": 580},
        ],
        "label": "high_risk",
    },
]


def _apply_gt_rules(row: pd.Series) -> str:
    """Apply ground truth rules to a single row (first-match wins)."""
    if row["debt_ratio"] > 0.6 and row["income"] < 40000:
        return "high_risk"
    if row["loan_amount"] > 15000 and row["employment_years"] < 2:
        return "high_risk"
    if row["credit_score"] < 580:
        return "high_risk"
    return "low_risk"


def generate(seed: int = 42, n: int = 500, noise_rate: float = 0.05):
    """Return (train_df, test_df) with an 80/20 split."""
    rng = np.random.default_rng(seed)

    income = rng.normal(55_000, 20_000, n).clip(10_000, 150_000).round(0)
    debt = rng.uniform(1_000, 80_000, n).round(0)
    debt_ratio = (debt / income).round(4)
    loan_amount = rng.uniform(2_000, 30_000, n).round(0)
    employment_years = rng.uniform(0, 40, n).round(1)
    credit_score = rng.normal(680, 80, n).clip(300, 850).round(0).astype(int)
    age = rng.integers(18, 71, n)
    random_feature_1 = rng.uniform(0, 1, n).round(4)
    irrelevant_score = rng.normal(0, 1, n).round(4)

    df = pd.DataFrame(
        {
            "income": income,
            "debt": debt,
            "debt_ratio": debt_ratio,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "credit_score": credit_score,
            "age": age,
            "random_feature_1": random_feature_1,
            "irrelevant_score": irrelevant_score,
        }
    )

    # Generate labels from ground truth rules (before adding any NaN)
    df["label"] = df.apply(_apply_gt_rules, axis=1)

    # Add ~5% label noise
    noise_mask = rng.random(n) < noise_rate
    df.loc[noise_mask, "label"] = df.loc[noise_mask, "label"].map(
        {"high_risk": "low_risk", "low_risk": "high_risk"}
    )

    # 80/20 split
    split = int(n * 0.8)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # Add ~10% missing values in employment_years in TRAIN only (after label generation)
    missing_mask = rng.random(split) < 0.10
    train_df.loc[missing_mask, "employment_years"] = float("nan")

    return train_df, test_df


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    train_df, test_df = generate()

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump({"rules": GROUND_TRUTH_RULES}, f, indent=2)

    print(f"Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    print(f"Train label distribution:\n{train_df['label'].value_counts().to_string()}")
    print(f"Test  label distribution:\n{test_df['label'].value_counts().to_string()}")
    print(f"Train missing in employment_years: {train_df['employment_years'].isna().sum()}")
    print(f"\nFiles written to {out_dir}")
