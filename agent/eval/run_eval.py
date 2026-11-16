"""
Evaluation orchestrator: runs each system N times, computes all metrics, saves results.

Usage:
  python eval/run_eval.py                        # 5 runs, budget=12, all systems
  python eval/run_eval.py --runs 3 --systems A,B # 3 runs for systems A and B only
  python eval/run_eval.py --model claude-sonnet-4-6  # use Sonnet for final eval

Environment:
  ANTHROPIC_API_KEY — required (set in your shell or .env)
  LLM_MODEL         — optional override (default: claude-haiku-4-5-20251001)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path regardless of invocation directory
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from eval.metrics import (
    compute_complexity,
    compute_f1,
    compute_faithfulness,
    compute_stability,
)
from systems import system_a, system_b, system_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation across all systems.")
    parser.add_argument("--runs", type=int, default=5, help="Runs per system (default: 5)")
    parser.add_argument("--budget", type=int, default=12, help="Max LLM calls per run (default: 12)")
    parser.add_argument("--systems", type=str, default="A,B,C", help="Comma-separated systems to run")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    args = parser.parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model

    # Paths
    train_path = _ROOT / "data" / "train.csv"
    test_path = _ROOT / "data" / "test.csv"
    gt_path = _ROOT / "data" / "ground_truth.json"
    logs_dir = _ROOT / "logs"
    results_dir = _ROOT / "results"
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Verify dataset exists
    if not train_path.exists():
        print("Dataset not found. Run first:\n  python data/generate.py")
        sys.exit(1)

    # Load evaluation data
    test_df = pd.read_csv(test_path)
    with open(gt_path) as f:
        gt_rules = json.load(f)["rules"]

    systems_map = {"A": system_a, "B": system_b, "C": system_c}
    selected = [s.strip().upper() for s in args.systems.split(",")]

    all_rows = []

    for sys_name in selected:
        if sys_name not in systems_map:
            print(f"Unknown system '{sys_name}', skipping.")
            continue

        mod = systems_map[sys_name]
        f1_scores = []

        print(f"\n{'='*50}")
        print(f"System {sys_name}  ({args.runs} runs, budget={args.budget})")
        print("=" * 50)

        for run_id in range(1, args.runs + 1):
            print(f"  Run {run_id}/{args.runs} ... ", end="", flush=True)
            t0 = time.time()

            try:
                result = mod.run(str(train_path), budget=args.budget)
            except Exception as exc:
                print(f"FAILED: {exc}")
                result = {
                    "output": None,
                    "log": {"system": sys_name, "error": str(exc), "llm_calls": [], "tool_calls": []},
                }

            elapsed = round(time.time() - t0, 1)

            # Persist log
            log_path = logs_dir / f"system_{sys_name}_run{run_id:02d}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(result["log"], f, indent=2, default=str)

            rules = result["output"]
            f1 = compute_f1(rules, test_df) if rules else 0.0
            n_rules, avg_conds = compute_complexity(rules) if rules else (0, 0.0)
            faith = compute_faithfulness(rules, gt_rules) if rules else 0.0
            f1_scores.append(f1)

            total_calls = result["log"].get(
                "total_calls", len(result["log"].get("llm_calls", []))
            )
            total_in = sum(c.get("input_tokens", 0) for c in result["log"].get("llm_calls", []))
            total_out = sum(c.get("output_tokens", 0) for c in result["log"].get("llm_calls", []))

            row = {
                "system": sys_name,
                "run": run_id,
                "f1": round(f1, 4),
                "n_rules": n_rules,
                "avg_conditions": avg_conds,
                "faithfulness": faith,
                "parse_success": rules is not None,
                "total_llm_calls": total_calls,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "elapsed_s": elapsed,
            }
            all_rows.append(row)

            status = f"F1={f1:.3f}  rules={n_rules}  faith={faith:.2f}  calls={total_calls}  ({elapsed}s)"
            if rules is None:
                status = f"PARSE FAILED  calls={total_calls}  ({elapsed}s)"
            print(status)

        stability = compute_stability(f1_scores)
        print(f"  → Stability (F1 std across {args.runs} runs): {stability}")

    # Save and display summary
    results_df = pd.DataFrame(all_rows)
    results_path = results_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*50}")
    print("SUMMARY (mean ± std across runs)")
    print("=" * 50)

    summary = (
        results_df.groupby("system")
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            stability=("f1", lambda x: compute_stability(x.tolist())),
            avg_rules=("n_rules", "mean"),
            avg_conditions=("avg_conditions", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            parse_rate=("parse_success", "mean"),
            avg_llm_calls=("total_llm_calls", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())
    print(f"\nFull results: {results_path}")
    print(f"Run logs:     {logs_dir}/")


if __name__ == "__main__":
    main()
