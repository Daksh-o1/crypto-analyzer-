"""
Phase 5 Baseline Models.

Implements five baseline models for the CryptoAnalyzer project:

1. NaiveBaseline         — lag-based deterministic benchmark
2. RidgeRegressorModel   — Ridge Regression for R(t,12)
3. LogisticModel         — Logistic Regression for D(t,12)
4. RandomForestModel     — RF for both regression and classification
5. XGBoostModel          — XGBoost for both regression and classification

Design Principles
-----------------
- All models receive 2D-flattened inputs: (N, 60*K) from the (N, 60, K) Phase 4 tensors.
- No model is ever fitted on validation or test data.
- Random seeds are fixed at RANDOM_SEED = 42 for full reproducibility.
- Hyperparameters are explicitly documented and not tuned against test metrics.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
REG_TARGET_COL: int = 0   # Index 0 in Y matrix = R_t_12
CLS_TARGET_COL: int = 1   # Index 1 in Y matrix = D_t_12


# ── Helper utilities ──────────────────────────────────────────────────────────

def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D sequence tensor (N, W, K) → 2D matrix (N, W*K).

    Flattening is done row-major (C-order), so for each sample the
    chronological ordering within the window is preserved:
      [t-59, t-58, ..., t]_feature_0, [t-59, ...]_feature_1, ...

    Parameters
    ----------
    X : np.ndarray, shape (N, W, K)

    Returns
    -------
    np.ndarray, shape (N, W*K)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")
    N, W, K = X.shape
    return X.reshape(N, W * K)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and a guarded MAPE for regression targets.

    MAPE is excluded when actual values are within 1e-8 of zero to prevent
    division-by-zero or astronomically large values from distorting the mean.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)

    Returns
    -------
    Dict with keys: 'mae', 'rmse', 'mape', 'mape_coverage'
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Guarded MAPE: only over rows where |y_true| > threshold
    threshold = 1e-8
    valid_mask = np.abs(y_true) > threshold
    mape_coverage = float(valid_mask.mean())
    if valid_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100)
    else:
        mape = float("nan")

    return {"mae": float(mae), "rmse": float(rmse), "mape": mape, "mape_coverage": mape_coverage}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute directional accuracy, precision, recall, macro-F1, and confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) — binary {0, 1}
    y_pred : np.ndarray, shape (N,) — binary {0, 1}

    Returns
    -------
    Dict with keys: 'accuracy', 'precision', 'recall', 'f1_macro', 'confusion_matrix'
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ── Model 1: Naive / Lag Baseline ─────────────────────────────────────────────

class NaiveBaseline:
    """
    Deterministic lag-based benchmark requiring no fitting.

    Regression:
        Predicts R(t, 12) ≈ the most-recent simple return in the input window.
        Specifically: uses the return computed from the last two close-price-
        equivalent values in the first feature column (index 0, which is 'open'
        in EXP_A or scaled price features).  For a fully scaled input this is
        a direction-preserving signal.

        Formally: pred_reg[i] = X[i, -1, 0] - X[i, -2, 0]
        (difference of the last two time-steps of feature 0)

    Classification:
        pred_cls[i] = 1 if pred_reg[i] > 0 else 0

    This baseline captures the hypothesis "recent momentum persists one hour"
    without any optimisation and without using future information.
    """

    def __init__(self):
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "NaiveBaseline":
        """No-op fit — naive baseline is deterministic."""
        self.is_fitted = True
        return self

    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Predict returns as the momentum signal from the last two steps."""
        if X.ndim == 2:
            # Already flattened — cannot reconstruct W without K; refuse
            raise ValueError("NaiveBaseline requires 3D input (N, W, K), not flattened.")
        # Momentum of feature 0 (typically close or open, scaled)
        return X[:, -1, 0] - X[:, -2, 0]

    def predict_classification(self, X: np.ndarray) -> np.ndarray:
        """Predict direction from momentum signal."""
        reg = self.predict_regression(X)
        return (reg > 0).astype(int)


# ── Model 2: Ridge Regression ─────────────────────────────────────────────────

class RidgeRegressorModel:
    """
    Ridge Regression for predicting R(t, 12).

    Hyperparameters (documented, not test-tuned):
    - alpha = 1.0   (L2 regularisation; default, mild shrinkage)
    - fit_intercept = True
    - random_state: not applicable (Ridge is deterministic)

    Input: flattened (N, W*K)
    Output: (N,) predicted percentage returns
    """

    HYPERPARAMS = {"alpha": 1.0, "fit_intercept": True}

    def __init__(self):
        self.model = Ridge(**self.HYPERPARAMS)
        self.is_fitted = False

    def fit(self, X_flat: np.ndarray, Y: np.ndarray) -> "RidgeRegressorModel":
        y_reg = Y[:, REG_TARGET_COL]
        self.model.fit(X_flat, y_reg)
        self.is_fitted = True
        return self

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)


# ── Model 3: Logistic Regression ──────────────────────────────────────────────

class LogisticModel:
    """
    Logistic Regression for predicting D(t, 12).

    Hyperparameters (documented, not test-tuned):
    - C = 0.1          (inverse regularisation; mild regularisation)
    - max_iter = 1000  (sufficient for convergence on ~6k samples)
    - solver = 'lbfgs'
    - random_state = 42

    Input: flattened (N, W*K)
    Output: (N,) binary predictions {0, 1}
    """

    HYPERPARAMS = {
        "C": 0.1,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    }

    def __init__(self):
        self.model = LogisticRegression(**self.HYPERPARAMS)
        self.is_fitted = False

    def fit(self, X_flat: np.ndarray, Y: np.ndarray) -> "LogisticModel":
        y_cls = Y[:, CLS_TARGET_COL].astype(int)
        self.model.fit(X_flat, y_cls)
        self.is_fitted = True
        return self

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)

    def predict_proba(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_flat)


# ── Model 4: Random Forest ────────────────────────────────────────────────────

class RandomForestModel:
    """
    Random Forest for both regression (R_t_12) and classification (D_t_12).

    Hyperparameters (documented, not test-tuned):
    - n_estimators = 100    (standard ensemble size)
    - max_depth = None      (fully grown trees; regularised by min_samples_leaf)
    - min_samples_leaf = 5  (guards against memorising single-sample leaves)
    - random_state = 42

    Input: flattened (N, W*K)
    """

    HYPERPARAMS = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 5,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    def __init__(self):
        self.reg_model = RandomForestRegressor(**self.HYPERPARAMS)
        self.cls_model = RandomForestClassifier(**self.HYPERPARAMS)
        self.is_fitted = False

    def fit(self, X_flat: np.ndarray, Y: np.ndarray) -> "RandomForestModel":
        y_reg = Y[:, REG_TARGET_COL]
        y_cls = Y[:, CLS_TARGET_COL].astype(int)
        self.reg_model.fit(X_flat, y_reg)
        self.cls_model.fit(X_flat, y_cls)
        self.is_fitted = True
        return self

    def predict_regression(self, X_flat: np.ndarray) -> np.ndarray:
        return self.reg_model.predict(X_flat)

    def predict_classification(self, X_flat: np.ndarray) -> np.ndarray:
        return self.cls_model.predict(X_flat)


# ── Model 5: XGBoost ─────────────────────────────────────────────────────────

class XGBoostModel:
    """
    XGBoost for both regression and classification.

    Requires: xgboost >= 2.0.0

    Hyperparameters (documented, not test-tuned):
    - n_estimators = 200
    - max_depth = 4
    - learning_rate = 0.05
    - subsample = 0.8
    - colsample_bytree = 0.8
    - random_state = 42
    - tree_method = 'hist'  (fast histogram method; CPU-based)
    - eval_metric: 'rmse' for regression, 'logloss' for classification

    Input: flattened (N, W*K)
    """

    HYPERPARAMS = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "verbosity": 0,
    }

    def __init__(self):
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Run: "
                "python -m pip install 'xgboost>=2.0.0'"
            )
        self.reg_model = xgb.XGBRegressor(eval_metric="rmse", **self.HYPERPARAMS)
        self.cls_model = xgb.XGBClassifier(eval_metric="logloss", **self.HYPERPARAMS)
        self.is_fitted = False

    def fit(self, X_flat: np.ndarray, Y: np.ndarray) -> "XGBoostModel":
        y_reg = Y[:, REG_TARGET_COL]
        y_cls = Y[:, CLS_TARGET_COL].astype(int)
        self.reg_model.fit(X_flat, y_reg)
        self.cls_model.fit(X_flat, y_cls)
        self.is_fitted = True
        return self

    def predict_regression(self, X_flat: np.ndarray) -> np.ndarray:
        return self.reg_model.predict(X_flat)

    def predict_classification(self, X_flat: np.ndarray) -> np.ndarray:
        return self.cls_model.predict(X_flat)
