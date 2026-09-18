"""
Phase 5 experiment runner.

Coordinates loading Phase 4 tensors, running all baseline models across
all four experiment configurations, evaluating on val and test splits,
and saving results as structured JSON artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from crypto_analyzer.models.baselines import (
    NaiveBaseline,
    RidgeRegressorModel,
    LogisticModel,
    RandomForestModel,
    XGBoostModel,
    flatten_sequences,
    compute_regression_metrics,
    compute_classification_metrics,
    _XGBOOST_AVAILABLE,
)

TENSOR_DIR = Path("D:/CryptoAnalyzer/data/processed/tensors")
RESULTS_DIR = Path("D:/CryptoAnalyzer/experiments/results")

EXPERIMENT_IDS = [
    "EXP_A_PRICE",
    "EXP_B_PRICE_VOL",
    "EXP_C_TECH_IND",
    "EXP_D_FULL",
]


def load_tensors(experiment_id: str) -> Dict[str, np.ndarray]:
    """Load the Phase 4 .npz tensor file for a given experiment."""
    path = TENSOR_DIR / f"{experiment_id}_tensors.npz"
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _evaluate_model(
    model_name: str,
    experiment_id: str,
    tensors: Dict[str, np.ndarray],
    model_obj: Any,
    use_naive: bool = False,
) -> Dict:
    """
    Fit model on training data, evaluate on val and test.

    CRITICAL LEAKAGE GUARD:
    - model_obj.fit() is called ONLY with X_train, Y_train.
    - Validation and test splits are never passed to fit().
    - Test metrics are computed once at the end only for reporting.
    """
    X_train = tensors["X_train"]
    Y_train = tensors["Y_train"]
    X_val   = tensors["X_val"]
    Y_val   = tensors["Y_val"]
    X_test  = tensors["X_test"]
    Y_test  = tensors["Y_test"]

    N_train, W, K = X_train.shape
    flat_dim = W * K

    result: Dict[str, Any] = {
        "model": model_name,
        "experiment": experiment_id,
        "N_train": int(N_train),
        "N_val": int(len(X_val)),
        "N_test": int(len(X_test)),
        "W": int(W),
        "K": int(K),
        "flat_dim": int(flat_dim),
        "random_seed": 42,
        "fit_on": "train_only",
        "val_metrics": {},
        "test_metrics": {},
    }

    t0 = time.perf_counter()

    if use_naive:
        model_obj.fit(X_train, Y_train)
        # Naive needs 3D input for momentum signal
        reg_val  = model_obj.predict_regression(X_val)
        cls_val  = model_obj.predict_classification(X_val)
        reg_test = model_obj.predict_regression(X_test)
        cls_test = model_obj.predict_classification(X_test)
    else:
        # Flatten to 2D for traditional ML
        X_tr_flat  = flatten_sequences(X_train)
        X_val_flat = flatten_sequences(X_val)
        X_te_flat  = flatten_sequences(X_test)

        model_obj.fit(X_tr_flat, Y_train)

        if hasattr(model_obj, "predict"):
            # Single-target model (Ridge → reg, Logistic → cls)
            preds = model_obj.predict(X_val_flat)
            preds_test = model_obj.predict(X_te_flat)
            if isinstance(model_obj, RidgeRegressorModel):
                reg_val, cls_val   = preds, None
                reg_test, cls_test = preds_test, None
            else:
                reg_val, cls_val   = None, preds
                reg_test, cls_test = None, preds_test
        else:
            reg_val  = model_obj.predict_regression(X_val_flat)
            cls_val  = model_obj.predict_classification(X_val_flat)
            reg_test = model_obj.predict_regression(X_te_flat)
            cls_test = model_obj.predict_classification(X_te_flat)

    elapsed = time.perf_counter() - t0
    result["fit_predict_seconds"] = round(elapsed, 2)

    # Compute metrics
    Y_val_reg  = Y_val[:, 0]
    Y_val_cls  = Y_val[:, 1].astype(int)
    Y_test_reg = Y_test[:, 0]
    Y_test_cls = Y_test[:, 1].astype(int)

    if reg_val is not None:
        result["val_metrics"]["regression"] = compute_regression_metrics(Y_val_reg, reg_val)
        result["test_metrics"]["regression"] = compute_regression_metrics(Y_test_reg, reg_test)

    if cls_val is not None:
        result["val_metrics"]["classification"] = compute_classification_metrics(Y_val_cls, cls_val)
        result["test_metrics"]["classification"] = compute_classification_metrics(Y_test_cls, cls_test)

    return result


def run_all_baselines(
    experiment_ids: list = None,
    model_names: list = None,
    results_dir: Path = RESULTS_DIR,
) -> Dict:
    """
    Run the full baseline experiment matrix.

    Parameters
    ----------
    experiment_ids : list or None
        Subset of experiment IDs to run (default: all four).
    model_names : list or None
        Subset of model names to run (default: all five).
    results_dir : Path
        Directory where JSON results are saved.

    Returns
    -------
    Dict mapping (experiment_id, model_name) → result dict
    """
    if experiment_ids is None:
        experiment_ids = EXPERIMENT_IDS
    if model_names is None:
        model_names = ["naive", "ridge", "logistic", "random_forest", "xgboost"]

    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for exp_id in experiment_ids:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_id}")
        print(f"{'='*60}")

        tensors = load_tensors(exp_id)
        K = tensors["X_train"].shape[2]
        print(f"  Loaded tensors — K={K}, "
              f"Train={tensors['X_train'].shape[0]}, "
              f"Val={tensors['X_val'].shape[0]}, "
              f"Test={tensors['X_test'].shape[0]}")

        exp_results = {}

        for model_name in model_names:
            print(f"\n  [{model_name.upper()}]")

            if model_name == "naive":
                model_obj = NaiveBaseline()
                res = _evaluate_model(model_name, exp_id, tensors, model_obj, use_naive=True)
            elif model_name == "ridge":
                model_obj = RidgeRegressorModel()
                res = _evaluate_model(model_name, exp_id, tensors, model_obj)
            elif model_name == "logistic":
                model_obj = LogisticModel()
                res = _evaluate_model(model_name, exp_id, tensors, model_obj)
            elif model_name == "random_forest":
                model_obj = RandomForestModel()
                res = _evaluate_model(model_name, exp_id, tensors, model_obj)
            elif model_name == "xgboost":
                if not _XGBOOST_AVAILABLE:
                    print("  [SKIP] xgboost not installed.")
                    continue
                model_obj = XGBoostModel()
                res = _evaluate_model(model_name, exp_id, tensors, model_obj)
            else:
                print(f"  [SKIP] Unknown model: {model_name}")
                continue

            exp_results[model_name] = res
            _print_result_summary(res)

        all_results[exp_id] = exp_results

        # Save per-experiment results
        out_path = results_dir / f"{exp_id}_baselines.json"
        with open(out_path, "w") as f:
            json.dump(exp_results, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    # Save combined results
    combined_path = results_dir / "all_baselines_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved: {combined_path}")

    return all_results


def _print_result_summary(res: Dict) -> None:
    """Print a compact one-line summary of a model result."""
    vm = res.get("val_metrics", {})
    if "regression" in vm:
        r = vm["regression"]
        print(f"    Val  REG  — MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} "
              f"MAPE={r['mape']:.1f}% (cov={r['mape_coverage']:.2%})")
    if "classification" in vm:
        c = vm["classification"]
        print(f"    Val  CLS  — Acc={c['accuracy']:.4f} F1={c['f1_macro']:.4f} "
              f"Prec={c['precision']:.4f} Rec={c['recall']:.4f}")
    tm = res.get("test_metrics", {})
    if "regression" in tm:
        r = tm["regression"]
        print(f"    Test REG  — MAE={r['mae']:.4f} RMSE={r['rmse']:.4f}")
    if "classification" in tm:
        c = tm["classification"]
        print(f"    Test CLS  — Acc={c['accuracy']:.4f} F1={c['f1_macro']:.4f}")
    print(f"    Fit+Predict: {res.get('fit_predict_seconds', '?')}s")
