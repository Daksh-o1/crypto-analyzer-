"""
Phase 5 Baseline Models CLI.

Usage examples:
  # Run all experiments and all models
  python scripts/run_baselines.py

  # Run a specific experiment
  python scripts/run_baselines.py --experiment EXP_A_PRICE

  # Run specific models only
  python scripts/run_baselines.py --model ridge --model logistic

  # Run one experiment with one model
  python scripts/run_baselines.py --experiment EXP_D_FULL --model xgboost
"""

import sys
import argparse
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from crypto_analyzer.models.experiment_runner import (
    run_all_baselines,
    EXPERIMENT_IDS,
    RESULTS_DIR,
)
from crypto_analyzer.models.baselines import _XGBOOST_AVAILABLE


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CryptoAnalyzer Phase 5 — Baseline Model Runner"
    )
    parser.add_argument(
        "--experiment", action="append", dest="experiments",
        choices=EXPERIMENT_IDS,
        help="Experiment configuration(s) to run. Repeat for multiple.",
    )
    parser.add_argument(
        "--model", action="append", dest="models",
        choices=["naive", "ridge", "logistic", "random_forest", "xgboost"],
        help="Model(s) to run. Repeat for multiple.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    experiments = args.experiments or EXPERIMENT_IDS
    models = args.models or ["naive", "ridge", "logistic", "random_forest", "xgboost"]

    print("=" * 70)
    print("CryptoAnalyzer - Phase 5: Baseline Models")
    print("=" * 70)
    print(f"Experiments : {experiments}")
    print(f"Models      : {models}")
    print(f"XGBoost     : {'Available' if _XGBOOST_AVAILABLE else 'NOT INSTALLED'}")
    print(f"Results Dir : {RESULTS_DIR}")
    print()
    print("LEAKAGE POLICY:")
    print("  - Models fitted on train split only.")
    print("  - Validation used for result reporting only (no tuning in this run).")
    print("  - Test data evaluated once at the end, not used for model selection.")
    print("=" * 70)

    all_results = run_all_baselines(
        experiment_ids=experiments,
        model_names=models,
        results_dir=RESULTS_DIR,
    )

    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL RESULT SUMMARY (Test Set)")
    print("=" * 70)
    header = f"{'Experiment':<22} {'Model':<16} {'Task':<8} {'MAE':>8} {'RMSE':>8} {'Acc':>8} {'F1':>8}"
    print(header)
    print("-" * 70)
    for exp_id, exp_results in all_results.items():
        for model_name, res in exp_results.items():
            tm = res.get("test_metrics", {})
            if "regression" in tm:
                r = tm["regression"]
                print(f"{exp_id:<22} {model_name:<16} {'REG':<8} "
                      f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {'—':>8} {'—':>8}")
            if "classification" in tm:
                c = tm["classification"]
                print(f"{exp_id:<22} {model_name:<16} {'CLS':<8} "
                      f"{'—':>8} {'—':>8} {c['accuracy']:>8.4f} {c['f1_macro']:>8.4f}")

    print("\n[PHASE GATE] Phase 5 COMPLETE. No Phase 6 work started.")


if __name__ == "__main__":
    main()
