"""Deep Learning Metric Computation Module."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_dl_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
) -> Dict[str, Any]:
    """Compute regression or classification evaluation metrics.

    Handles zero/near-zero division in MAPE safely matching Phase 5 implementation.

    For classification, both binary (positive-class) F1 and macro-averaged F1 are
    returned. The research specification requires macro F1 for cross-model comparison.
    Binary F1 is preserved for auditability and cross-reference with earlier runs.

    Classification output keys
    --------------------------
    accuracy       : overall accuracy
    precision      : positive-class precision (binary, label=1)
    recall         : positive-class recall (binary, label=1)
    f1_binary      : positive-class F1 (sklearn default, average='binary')
    f1_macro       : macro-averaged F1 (average='macro') — research-spec metric
    confusion_matrix: [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if task == "regression":
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # Safe MAPE computation matching Phase 5 baseline handling
        abs_true = np.abs(y_true)
        valid_mask = abs_true > 1e-8
        if np.any(valid_mask):
            mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100.0)
        else:
            mape = 0.0

        return {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
        }

    elif task == "classification":
        # Convert logits/probabilities or continuous predictions to binary 0/1
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)

        acc = float(accuracy_score(y_true_bin, y_pred_bin))
        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        # Binary (positive-class) F1 — preserved from original implementation
        f1_bin = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        # Macro F1 — required by research specification (matches Phase 5 baselines.py)
        f1_mac = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
        cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()

        return {
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_binary": round(f1_bin, 6),
            "f1_macro": round(f1_mac, 6),
            "confusion_matrix": cm,
        }

    else:
        raise ValueError(f"Unknown task '{task}'. Must be 'regression' or 'classification'.")

