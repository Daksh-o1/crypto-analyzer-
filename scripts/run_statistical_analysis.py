"""
Phase 9 Script: Statistical Significance & Comparative Analysis

Performs Wilcoxon signed-rank tests with Holm correction on targeted pairs.
"""

import sys
import json
import warnings
import gc
from pathlib import Path
import numpy as np
import scipy.stats

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crypto_analyzer.models.baselines import (
    NaiveBaseline, RidgeRegressorModel, LogisticModel,
    RandomForestModel, XGBoostModel, flatten_sequences,
    _XGBOOST_AVAILABLE
)

ROOT_DIR = Path("D:/CryptoAnalyzer")
TENSOR_DIR = ROOT_DIR / "data/processed/tensors"
CHECKPOINT_DIR = ROOT_DIR / "models/checkpoints"
RESULTS_DIR = ROOT_DIR / "experiments/results"
TABLES_DIR = ROOT_DIR / "reports/tables"

def load_tensors(exp_id):
    path = TENSOR_DIR / f"{exp_id}_tensors.npz"
    return np.load(path)

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

def dl_evaluate(model, loader):
    import torch
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            out = model(batch_x.to(torch.float32))
            all_preds.append(out.cpu().numpy().ravel())
    return np.concatenate(all_preds)

def holm_correction(p_values):
    """Applies Holm step-down correction for multiple comparisons."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adj = p_values[idx] * (n - i)
        if i > 0:
            adj = max(adj, adjusted_p[sorted_indices[i - 1]])
        adjusted_p[idx] = min(adj, 1.0)
    return adjusted_p

def run_wilcoxon(pred1, pred2, y_true, task):
    if task == "regression":
        err1 = np.abs(y_true - pred1)
        err2 = np.abs(y_true - pred2)
        diff = err1 - err2
    else:
        # For classification, hit series (1 correct, 0 incorrect)
        hit1 = (y_true == pred1).astype(int)
        hit2 = (y_true == pred2).astype(int)
        diff = hit1 - hit2
        
    n_total = len(diff)
    n_nonzero = np.sum(diff != 0)
    
    if n_nonzero == 0:
        return {"statistic": float('nan'), "p_value": 1.0, "n_total": n_total, "n_nonzero": 0, "mean_diff": 0.0}
        
    # Wilcoxon signed-rank test
    stat, p = scipy.stats.wilcoxon(diff, zero_method='zsplit', alternative='two-sided')
    mean_diff = np.mean(diff)
    
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_total": int(n_total),
        "n_nonzero": int(n_nonzero),
        "mean_diff": float(mean_diff)
    }

def get_baseline_preds():
    """Extracts required baseline predictions sequentially, ensuring n_jobs=-1 works before torch loads."""
    preds = {}
    
    # Needs: XGBoost (A, B, C, D), Naive/Ridge/Logistic/RF (D only)
    for exp_id in ["EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"]:
        tensors = load_tensors(exp_id)
        X_train_flat = flatten_sequences(tensors["X_train"])
        X_test_flat = flatten_sequences(tensors["X_test"])
        
        # XGBoost on all
        print(f"[{exp_id}] Fitting XGBoost...")
        xgb = XGBoostModel()
        xgb.fit(X_train_flat, tensors["Y_train"])
        preds[f"{exp_id}_xgboost_reg"] = xgb.predict_regression(X_test_flat)
        preds[f"{exp_id}_xgboost_cls"] = xgb.predict_classification(X_test_flat)
        del xgb
        
        if exp_id == "EXP_D_FULL":
            print(f"[{exp_id}] Fitting Naive...")
            naive = NaiveBaseline()
            naive.fit(tensors["X_train"], tensors["Y_train"])
            preds[f"{exp_id}_naive_reg"] = naive.predict_regression(tensors["X_test"])
            preds[f"{exp_id}_naive_cls"] = naive.predict_classification(tensors["X_test"])
            
            print(f"[{exp_id}] Fitting Ridge/Logistic...")
            ridge = RidgeRegressorModel()
            logi = LogisticModel()
            ridge.fit(X_train_flat, tensors["Y_train"])
            logi.fit(X_train_flat, tensors["Y_train"])
            preds[f"{exp_id}_ridge_reg"] = ridge.predict(X_test_flat)
            preds[f"{exp_id}_logistic_cls"] = logi.predict(X_test_flat)
            
            print(f"[{exp_id}] Fitting Random Forest...")
            rf = RandomForestModel()
            rf.HYPERPARAMS["n_jobs"] = -1
            rf.fit(X_train_flat, tensors["Y_train"])
            preds[f"{exp_id}_random_forest_reg"] = rf.predict_regression(X_test_flat)
            preds[f"{exp_id}_random_forest_cls"] = rf.predict_classification(X_test_flat)
            del rf
            
        del tensors, X_train_flat, X_test_flat
        gc.collect()
        
    return preds

def get_dl_preds():
    """Extracts required DL predictions after PyTorch is imported."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from crypto_analyzer.models.dl.trainer import set_seed
    from crypto_analyzer.models.dl.checkpoint import load_checkpoint
    
    set_seed(42)
    torch.set_num_threads(1)
    
    preds = {}
    
    # Needs: LSTM (A, B, C, D), GRU/1D-CNN (D only)
    for exp_id in ["EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"]:
        tensors = load_tensors(exp_id)
        input_size = tensors["X_test"].shape[2]
        
        y_test_reg = tensors["Y_test"][:, 0]
        y_test_cls = tensors["Y_test"][:, 1].astype(int)
        
        loader_reg = DataLoader(TensorDataset(torch.tensor(tensors["X_test"], dtype=torch.float32), torch.tensor(y_test_reg, dtype=torch.float32)), batch_size=64)
        loader_cls = DataLoader(TensorDataset(torch.tensor(tensors["X_test"], dtype=torch.float32), torch.tensor(y_test_cls, dtype=torch.float32)), batch_size=64)
        
        models_to_run = ["LSTM"]
        if exp_id == "EXP_D_FULL":
            models_to_run.extend(["GRU", "1D-CNN"])
            
        for arch in models_to_run:
            arch_tag = arch.lower().replace("-", "")
            
            # Reg
            ckpt = CHECKPOINT_DIR / f"{arch_tag}_{exp_id}_regression.pt"
            model_reg = build_dl_model(arch, input_size, "regression")
            load_checkpoint(model_reg, ckpt, device=torch.device("cpu"))
            preds[f"{exp_id}_{arch}_reg"] = dl_evaluate(model_reg, loader_reg)
            
            # Cls
            ckpt_cls = CHECKPOINT_DIR / f"{arch_tag}_{exp_id}_classification.pt"
            model_cls = build_dl_model(arch, input_size, "classification")
            load_checkpoint(model_cls, ckpt_cls, device=torch.device("cpu"))
            preds[f"{exp_id}_{arch}_cls"] = dl_evaluate(model_cls, loader_cls)
            
        del tensors, loader_reg, loader_cls
        gc.collect()
        
    return preds

def main():
    print("Generating baseline predictions...")
    preds = get_baseline_preds()
    
    print("Generating DL predictions...")
    dl_preds = get_dl_preds()
    preds.update(dl_preds)
    
    # Load targets (they are the same across all experiments, so just load D)
    tensors = load_tensors("EXP_D_FULL")
    y_test_reg = tensors["Y_test"][:, 0]
    y_test_cls = tensors["Y_test"][:, 1].astype(int)
    
    comparisons = [
        # Regression Feature
        ("regression", "XGBoost: EXP_A vs EXP_B", "EXP_A_PRICE_xgboost_reg", "EXP_B_PRICE_VOL_xgboost_reg"),
        ("regression", "XGBoost: EXP_A vs EXP_C", "EXP_A_PRICE_xgboost_reg", "EXP_C_TECH_IND_xgboost_reg"),
        ("regression", "XGBoost: EXP_A vs EXP_D", "EXP_A_PRICE_xgboost_reg", "EXP_D_FULL_xgboost_reg"),
        ("regression", "LSTM: EXP_A vs EXP_B", "EXP_A_PRICE_LSTM_reg", "EXP_B_PRICE_VOL_LSTM_reg"),
        ("regression", "LSTM: EXP_A vs EXP_C", "EXP_A_PRICE_LSTM_reg", "EXP_C_TECH_IND_LSTM_reg"),
        ("regression", "LSTM: EXP_A vs EXP_D", "EXP_A_PRICE_LSTM_reg", "EXP_D_FULL_LSTM_reg"),
        
        # Regression Model
        ("regression", "Naive vs Ridge (EXP_D)", "EXP_D_FULL_naive_reg", "EXP_D_FULL_ridge_reg"),
        ("regression", "Ridge vs Random Forest (EXP_D)", "EXP_D_FULL_ridge_reg", "EXP_D_FULL_random_forest_reg"),
        ("regression", "Random Forest vs XGBoost (EXP_D)", "EXP_D_FULL_random_forest_reg", "EXP_D_FULL_xgboost_reg"),
        ("regression", "XGBoost vs LSTM (EXP_D)", "EXP_D_FULL_xgboost_reg", "EXP_D_FULL_LSTM_reg"),
        ("regression", "LSTM vs GRU (EXP_D)", "EXP_D_FULL_LSTM_reg", "EXP_D_FULL_GRU_reg"),
        ("regression", "GRU vs 1D-CNN (EXP_D)", "EXP_D_FULL_GRU_reg", "EXP_D_FULL_1D-CNN_reg"),
        
        # Classification Feature
        ("classification", "XGBoost: EXP_A vs EXP_B", "EXP_A_PRICE_xgboost_cls", "EXP_B_PRICE_VOL_xgboost_cls"),
        ("classification", "XGBoost: EXP_A vs EXP_C", "EXP_A_PRICE_xgboost_cls", "EXP_C_TECH_IND_xgboost_cls"),
        ("classification", "XGBoost: EXP_A vs EXP_D", "EXP_A_PRICE_xgboost_cls", "EXP_D_FULL_xgboost_cls"),
        ("classification", "LSTM: EXP_A vs EXP_B", "EXP_A_PRICE_LSTM_cls", "EXP_B_PRICE_VOL_LSTM_cls"),
        ("classification", "LSTM: EXP_A vs EXP_C", "EXP_A_PRICE_LSTM_cls", "EXP_C_TECH_IND_LSTM_cls"),
        ("classification", "LSTM: EXP_A vs EXP_D", "EXP_A_PRICE_LSTM_cls", "EXP_D_FULL_LSTM_cls"),
        
        # Classification Model
        ("classification", "Naive vs Logistic (EXP_D)", "EXP_D_FULL_naive_cls", "EXP_D_FULL_logistic_cls"),
        ("classification", "Logistic vs Random Forest (EXP_D)", "EXP_D_FULL_logistic_cls", "EXP_D_FULL_random_forest_cls"),
        ("classification", "Random Forest vs XGBoost (EXP_D)", "EXP_D_FULL_random_forest_cls", "EXP_D_FULL_xgboost_cls"),
        ("classification", "XGBoost vs LSTM (EXP_D)", "EXP_D_FULL_xgboost_cls", "EXP_D_FULL_LSTM_cls"),
        ("classification", "LSTM vs GRU (EXP_D)", "EXP_D_FULL_LSTM_cls", "EXP_D_FULL_GRU_cls"),
        ("classification", "GRU vs 1D-CNN (EXP_D)", "EXP_D_FULL_GRU_cls", "EXP_D_FULL_1D-CNN_cls")
    ]
    
    results = []
    reg_pvals, cls_pvals = [], []
    
    for task, comp, k1, k2 in comparisons:
        y_true = y_test_reg if task == "regression" else y_test_cls
        res = run_wilcoxon(preds[k1], preds[k2], y_true, task)
        res["task"] = task
        res["comparison"] = comp
        results.append(res)
        
        if task == "regression":
            reg_pvals.append(res["p_value"])
        else:
            cls_pvals.append(res["p_value"])
            
    # Apply Holm correction separately per task
    reg_adj = holm_correction(reg_pvals)
    cls_adj = holm_correction(cls_pvals)
    
    r_idx, c_idx = 0, 0
    for res in results:
        if res["task"] == "regression":
            res["adjusted_p_value"] = reg_adj[r_idx]
            r_idx += 1
        else:
            res["adjusted_p_value"] = cls_adj[c_idx]
            c_idx += 1
            
        res["significant"] = bool(res["adjusted_p_value"] < 0.05)
        
    output_data = {
        "metadata": {
            "methodology": "Paired Wilcoxon signed-rank test (two-sided, Holm step-down correction)",
            "significance_threshold": 0.05,
            "test_sample_size": 1282,
            "regression_metric": "Absolute Prediction Error |y_true - y_pred|",
            "classification_metric": "Hit Series (1=correct, 0=incorrect, discrete {-1,0,1} difference)"
        },
        "results": results
    }
    
    with open(RESULTS_DIR / "statistical_significance.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    # Write Markdown Tables
    for task in ["regression", "classification"]:
        lines = [
            f"# Phase 9 Statistical Significance: {task.capitalize()}\n",
            "**Method:** Two-sided paired Wilcoxon signed-rank test. $p$-values adjusted via Holm step-down correction.\n",
            "**Note:** Statistical significance ($\alpha = 0.05$) denotes that the median of differences is non-zero, not necessarily a large practical effect or 'best' model.\n",
            "| Comparison | N (total) | N (non-zero) | Statistic (W) | Mean Diff | Raw $p$-value | Adj $p$-value | Significant |",
            "|---|---|---|---|---|---|---|---|"
        ]
        
        for r in results:
            if r["task"] == task:
                diff_str = f"{r['mean_diff']:.4f}"
                sig_str = "**YES**" if r["significant"] else "No"
                line = f"| {r['comparison']} | {r['n_total']} | {r['n_nonzero']} | {r['statistic']:.2f} | {diff_str} | {r['p_value']:.4e} | {r['adjusted_p_value']:.4e} | {sig_str} |"
                lines.append(line)
                
        with open(TABLES_DIR / f"statistical_significance_{task}.md", "w") as f:
            f.write("\n".join(lines))
            
    print("Done.")

if __name__ == "__main__":
    main()
