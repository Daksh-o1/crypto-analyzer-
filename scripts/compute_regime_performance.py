"""
Phase 8 Script: Compute Regime-wise Model Performance.

Reconstructs model predictions for the test partition without retraining:
- Baselines are deterministically re-fitted.
- DL models are loaded from existing Phase 6 checkpoints.
Validates reconstructed predictions against Phase 5/6 results within tolerance.
Computes and saves performance metrics for each regime slice.
"""

import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict
import gc

import numpy as np
import pandas as pd


# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crypto_analyzer.models.baselines import (
    NaiveBaseline, RidgeRegressorModel, LogisticModel,
    RandomForestModel, XGBoostModel, flatten_sequences,
    compute_regression_metrics, compute_classification_metrics,
    _XGBOOST_AVAILABLE
)


ROOT_DIR = Path("D:/CryptoAnalyzer")
TENSOR_DIR = ROOT_DIR / "data/processed/tensors"
CHECKPOINT_DIR = ROOT_DIR / "models/checkpoints"
RESULTS_DIR = ROOT_DIR / "experiments/results"
TABLES_DIR = ROOT_DIR / "reports/tables"

EXPERIMENTS = ["EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"]
BASELINE_MODELS = ["naive", "ridge", "logistic", "random_forest", "xgboost"]
DL_MODELS = ["LSTM", "GRU", "1D-CNN"]
REGIMES = ["Bullish", "Bearish", "Sideways", "High Volatility"]



TOLERANCE = 1e-3

def assert_close(name, old_val, new_val):
    if np.isnan(old_val) and np.isnan(new_val):
        return
    diff = abs(old_val - new_val)
    if diff >= TOLERANCE:
        raise ValueError(f"Metric '{name}' mismatch! Expected: {old_val}, Got: {new_val}")


def load_regime_labels():
    df = pd.read_csv(RESULTS_DIR / "test_regime_labels.csv")
    if len(df) != 1282:
        raise ValueError(f"Expected 1282 test regime labels, got {len(df)}")
    return df["regime"].values


def load_tensors(exp_name):
    data = np.load(TENSOR_DIR / f"{exp_name}_tensors.npz")
    return {k: data[k] for k in ["X_train", "Y_train", "X_test", "Y_test"]}


def dl_evaluate(model, loader):
    import torch
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            out = model(batch_x.to(torch.float32))
            all_preds.append(out.cpu().numpy().ravel())
            all_trues.append(batch_y.cpu().numpy().ravel())
    return np.concatenate(all_trues), np.concatenate(all_preds)


def run_baseline_reconstruction(exp_id, tensors, model_name):
    model_obj = None
    if model_name == "naive": model_obj = NaiveBaseline()
    elif model_name == "ridge": model_obj = RidgeRegressorModel()
    elif model_name == "logistic": model_obj = LogisticModel()
    elif model_name == "random_forest":
        model_obj = RandomForestModel()
        # Force n_jobs=1 to prevent joblib/torch deadlock on Windows
        model_obj.reg_model.set_params(n_jobs=1)
        model_obj.cls_model.set_params(n_jobs=1)
    elif model_name == "xgboost":
        if not _XGBOOST_AVAILABLE: return None, None
        model_obj = XGBoostModel()
    
    if model_name == "naive":
        model_obj.fit(tensors["X_train"], tensors["Y_train"])
        reg_pred = model_obj.predict_regression(tensors["X_test"])
        cls_pred = model_obj.predict_classification(tensors["X_test"])
    else:
        X_train_flat = flatten_sequences(tensors["X_train"])
        X_test_flat = flatten_sequences(tensors["X_test"])
        model_obj.fit(X_train_flat, tensors["Y_train"])
        if hasattr(model_obj, "predict"):
            preds = model_obj.predict(X_test_flat)
            if model_name == "ridge":
                reg_pred, cls_pred = preds, None
            else:
                reg_pred, cls_pred = None, preds
        else:
            reg_pred = model_obj.predict_regression(X_test_flat)
            cls_pred = model_obj.predict_classification(X_test_flat)
    return reg_pred, cls_pred


def build_dl_model(arch_name, input_size, task):
    from crypto_analyzer.models.dl.config import DLConfig
    from crypto_analyzer.models.dl.lstm import LSTMModel
    from crypto_analyzer.models.dl.gru import GRUModel
    from crypto_analyzer.models.dl.cnn import CNN1DModel
    dl_config = DLConfig()
    dl_arch = {"LSTM": LSTMModel, "GRU": GRUModel, "1D-CNN": CNN1DModel}
    if arch_name == "1D-CNN":
        return CNN1DModel(input_size=input_size, channels=dl_config.cnn_channels, dropout=dl_config.dropout, task=task)
    return dl_arch[arch_name](input_size=input_size, hidden_size=dl_config.hidden_size, num_layers=dl_config.num_layers, dropout=dl_config.dropout, task=task)


def main():
    print("Reconstructing predictions and verifying aggregate metrics...")

    
    with open(RESULTS_DIR / "all_baselines_results.json") as f:
        baseline_results = json.load(f)
        
    test_regimes = load_regime_labels()
    
    regime_results = []
    
    for exp_id in EXPERIMENTS:
        tensors = load_tensors(exp_id)
        input_size = tensors["X_train"].shape[2]
        
        y_test_reg = tensors["Y_test"][:, 0]
        y_test_cls = tensors["Y_test"][:, 1].astype(int)
        
        # DL Results
        with open(RESULTS_DIR / f"{exp_id}_dl_results.json") as f:
            dl_res = json.load(f)
            
        # BASELINES
        for model_name in BASELINE_MODELS:
            print(f"[{exp_id}] Reconstructing {model_name}...")
            reg_pred, cls_pred = run_baseline_reconstruction(exp_id, tensors, model_name)
            if reg_pred is None and cls_pred is None:
                continue
            
            exp_results = baseline_results.get(exp_id, {}).get(model_name, {}).get("test_metrics", {})
            
            if reg_pred is not None:
                agg_metrics = compute_regression_metrics(y_test_reg, reg_pred)
                old_metrics = exp_results.get("regression", {})
                for k in ["mae", "rmse", "mape"]:
                    assert_close(f"{model_name} {exp_id} reg {k}", old_metrics[k], agg_metrics[k])
                
                # Regime slicing
                for r in REGIMES:
                    mask = (test_regimes == r)
                    count = int(mask.sum())
                    if count > 0:
                        m = compute_regression_metrics(y_test_reg[mask], reg_pred[mask])
                        regime_results.append({
                            "experiment": exp_id, "model": model_name, "task": "regression",
                            "regime": r, "sample_count": count, "metrics": m
                        })
                    else:
                        regime_results.append({
                            "experiment": exp_id, "model": model_name, "task": "regression",
                            "regime": r, "sample_count": count, "metrics": None
                        })
                        
            if cls_pred is not None:
                agg_metrics = compute_classification_metrics(y_test_cls, cls_pred)
                old_metrics = exp_results.get("classification", {})
                for k in ["accuracy", "precision", "recall", "f1_macro"]:
                    assert_close(f"{model_name} {exp_id} cls {k}", old_metrics[k], agg_metrics[k])
                
                # Regime slicing
                for r in REGIMES:
                    mask = (test_regimes == r)
                    count = int(mask.sum())
                    if count > 0:
                        m = compute_classification_metrics(y_test_cls[mask], cls_pred[mask])
                        regime_results.append({
                            "experiment": exp_id, "model": model_name, "task": "classification",
                            "regime": r, "sample_count": count, "metrics": m
                        })
                    else:
                        regime_results.append({
                            "experiment": exp_id, "model": model_name, "task": "classification",
                            "regime": r, "sample_count": count, "metrics": None
                        })
                        
            del reg_pred, cls_pred
            gc.collect()
                        
        # DEEP LEARNING
        for arch in DL_MODELS:
            print(f"[{exp_id}] Reconstructing {arch}...")
            arch_tag = arch.lower().replace("-", "")
            
            import torch
            from crypto_analyzer.models.dl.trainer import set_seed
            set_seed(42)
            torch.set_num_threads(1)
            from torch.utils.data import TensorDataset, DataLoader
            from crypto_analyzer.models.dl.checkpoint import load_checkpoint
            from crypto_analyzer.models.dl.metrics import compute_dl_metrics
            
            # Regression
            ckpt = CHECKPOINT_DIR / f"{arch_tag}_{exp_id}_regression.pt"
            model_reg = build_dl_model(arch, input_size, "regression")
            load_checkpoint(model_reg, ckpt, device=torch.device("cpu"))
            _, reg_pred = dl_evaluate(model_reg, DataLoader(TensorDataset(torch.tensor(tensors["X_test"], dtype=torch.float32), torch.tensor(y_test_reg, dtype=torch.float32)), batch_size=64))
            
            agg_reg = compute_dl_metrics(y_test_reg, reg_pred, "regression")
            old_reg = dl_res[arch]["regression"]["metrics"]["test"]
            for k in ["mae", "rmse", "mape"]:
                assert_close(f"{arch} {exp_id} reg {k}", old_reg[k], agg_reg[k])
                
            for r in REGIMES:
                mask = (test_regimes == r)
                count = int(mask.sum())
                if count > 0:
                    m = compute_dl_metrics(y_test_reg[mask], reg_pred[mask], "regression")
                    regime_results.append({
                        "experiment": exp_id, "model": arch, "task": "regression",
                        "regime": r, "sample_count": count, "metrics": m
                    })
                else:
                    regime_results.append({
                        "experiment": exp_id, "model": arch, "task": "regression",
                        "regime": r, "sample_count": count, "metrics": None
                    })
                    
            # Classification
            ckpt_cls = CHECKPOINT_DIR / f"{arch_tag}_{exp_id}_classification.pt"
            model_cls = build_dl_model(arch, input_size, "classification")
            load_checkpoint(model_cls, ckpt_cls, device=torch.device("cpu"))
            _, cls_pred = dl_evaluate(model_cls, DataLoader(TensorDataset(torch.tensor(tensors["X_test"], dtype=torch.float32), torch.tensor(y_test_cls, dtype=torch.float32)), batch_size=64))
            
            agg_cls = compute_dl_metrics(y_test_cls, cls_pred, "classification")
            old_cls = dl_res[arch]["classification"]["metrics"]["test"]
            for k in ["accuracy", "precision", "recall", "f1_macro", "f1_binary"]:
                assert_close(f"{arch} {exp_id} cls {k}", old_cls[k], agg_cls[k])
                
            for r in REGIMES:
                mask = (test_regimes == r)
                count = int(mask.sum())
                if count > 0:
                    m = compute_dl_metrics(y_test_cls[mask], cls_pred[mask], "classification")
                    regime_results.append({
                        "experiment": exp_id, "model": arch, "task": "classification",
                        "regime": r, "sample_count": count, "metrics": m
                    })
                else:
                    regime_results.append({
                        "experiment": exp_id, "model": arch, "task": "classification",
                        "regime": r, "sample_count": count, "metrics": None
                    })

            del model_reg, model_cls, reg_pred, cls_pred
            gc.collect()

        del tensors, y_test_reg, y_test_cls
        gc.collect()

    # Save output
    output_data = {
        "metadata": {
            "methodology": "Phase 8 Regime Segmentation",
            "regime_window": 288,
            "theta_slope_documented_default": 0.0005,
            "theta_slope_used": 0.000040,
            "theta_vol_high": 0.001896,
            "partitions": {"train": "0-5979", "val": "5980-7261", "test": "7262-8543"}
        },
        "results": regime_results
    }
    
    with open(RESULTS_DIR / "regime_model_performance.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    # Generate Markdown Tables
    for task in ["regression", "classification"]:
        lines = [f"# Phase 8 Regime-wise Model Performance: {task.capitalize()}\n"]
        lines.append("| Experiment | Model | Regime | Sample Count | " + 
                     ("MAE | RMSE | MAPE |" if task == "regression" else "Accuracy | Precision | Recall | F1 Macro | F1 Binary |"))
        lines.append("|---|---|---|---|" + ("---|---|---|" if task == "regression" else "---|---|---|---|---|"))
        
        for r in regime_results:
            if r["task"] == task:
                m = r["metrics"]
                if m is None:
                    if task == "regression":
                        mets = "— | — | — |"
                    else:
                        mets = "— | — | — | — | — |"
                else:
                    if task == "regression":
                        mets = f"{m['mae']:.4f} | {m['rmse']:.4f} | {m['mape']:.2f}% |"
                    else:
                        mets = f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_macro']:.4f} | {m.get('f1_binary', np.nan):.4f} |"
                lines.append(f"| {r['experiment']} | {r['model']} | {r['regime']} | {r['sample_count']} | {mets}")
                
        with open(TABLES_DIR / f"regime_model_performance_{task}.md", "w") as f:
            f.write("\n".join(lines))
            
    print("Done. Saved regime_model_performance.json and markdown tables.")

if __name__ == "__main__":
    main()
