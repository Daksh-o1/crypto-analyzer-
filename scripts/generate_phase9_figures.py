"""
Phase 9 Script: Generate Missing Figures

Generates:
1. ROC curves
2. Confusion matrices
3. Prediction-vs-actual return scatter plots
"""

import sys
import gc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crypto_analyzer.models.baselines import (
    XGBoostModel, flatten_sequences
)

ROOT_DIR = Path("D:/CryptoAnalyzer")
TENSOR_DIR = ROOT_DIR / "data/processed/tensors"
CHECKPOINT_DIR = ROOT_DIR / "models/checkpoints"
FIGURES_DIR = ROOT_DIR / "reports/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_tensors(exp_id):
    return np.load(TENSOR_DIR / f"{exp_id}_tensors.npz")

def main():
    print("Loading EXP_D_FULL data...")
    tensors = load_tensors("EXP_D_FULL")
    X_train_flat = flatten_sequences(tensors["X_train"])
    X_test_flat = flatten_sequences(tensors["X_test"])
    y_test_reg = tensors["Y_test"][:, 0]
    y_test_cls = tensors["Y_test"][:, 1].astype(int)

    # 1. XGBoost
    print("Fitting XGBoost...")
    xgb = XGBoostModel()
    xgb.fit(X_train_flat, tensors["Y_train"])
    
    xgb_reg_pred = xgb.predict_regression(X_test_flat)
    xgb_cls_pred = xgb.predict_classification(X_test_flat)
    xgb_cls_prob = xgb.cls_model.predict_proba(X_test_flat)[:, 1]
    
    del xgb
    gc.collect()

    # 2. LSTM
    print("Loading LSTM...")
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from crypto_analyzer.models.dl.lstm import LSTMModel
    from crypto_analyzer.models.dl.checkpoint import load_checkpoint
    
    input_size = tensors["X_test"].shape[2]
    
    # LSTM Regression
    lstm_reg = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2, task="regression")
    load_checkpoint(lstm_reg, CHECKPOINT_DIR / "lstm_EXP_D_FULL_regression.pt", device=torch.device("cpu"))
    lstm_reg.eval()
    
    # LSTM Classification
    lstm_cls = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2, task="classification")
    load_checkpoint(lstm_cls, CHECKPOINT_DIR / "lstm_EXP_D_FULL_classification.pt", device=torch.device("cpu"))
    lstm_cls.eval()
    
    test_tensor = torch.tensor(tensors["X_test"], dtype=torch.float32)
    with torch.no_grad():
        lstm_reg_pred = lstm_reg(test_tensor).cpu().numpy().ravel()
        logits = lstm_cls(test_tensor).cpu().numpy().ravel()
        lstm_cls_prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        lstm_cls_pred = (lstm_cls_prob >= 0.5).astype(int)

    # Plot 1: Prediction vs Actual Scatter Plots (Regression)
    print("Generating Scatter Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(y_test_reg, xgb_reg_pred, alpha=0.3, color='blue', label='XGBoost')
    axes[0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', label='Perfect')
    axes[0].set_title('XGBoost: Predicted vs Actual Returns')
    axes[0].set_xlabel('Actual Return')
    axes[0].set_ylabel('Predicted Return')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(y_test_reg, lstm_reg_pred, alpha=0.3, color='orange', label='LSTM')
    axes[1].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', label='Perfect')
    axes[1].set_title('LSTM: Predicted vs Actual Returns')
    axes[1].set_xlabel('Actual Return')
    axes[1].set_ylabel('Predicted Return')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_vs_actual_scatter.png", dpi=300)
    plt.close()

    # Plot 2: Confusion Matrices (Classification)
    print("Generating Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_xgb = confusion_matrix(y_test_cls, xgb_cls_pred)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('XGBoost Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    cm_lstm = confusion_matrix(y_test_cls, lstm_cls_pred)
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('LSTM Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close()
    
    # Plot 3: ROC Curves (Classification)
    print("Generating ROC Curves...")
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test_cls, xgb_cls_prob)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test_cls, lstm_cls_prob)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
    plt.plot(fpr_lstm, tpr_lstm, color='navy', lw=2, label=f'LSTM (AUC = {roc_auc_lstm:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (EXP_D_FULL)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close()

    print("Missing Phase 9 figures generated successfully.")

if __name__ == "__main__":
    main()
