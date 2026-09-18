"""
Test suite for Phase 5 Baseline Models.

Tests cover:
- Model initialization
- Input flattening and dimensionality
- Train-only fitting contract
- Prediction shape and validity
- Classification output validity {0, 1}
- Regression output finiteness
- Reproducibility with fixed random seed
- Naive baseline correctness
- Metric computation correctness
- All four EXP_A/B/C/D configurations
- MAPE handling for near-zero returns

Uses small synthetic fixtures to avoid expensive full training in CI.
"""

import numpy as np
import pytest

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
    RANDOM_SEED,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_tensors(N: int, W: int, K: int, seed: int = 0):
    """Return synthetic 3D tensors and targets for unit testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, W, K)).astype(np.float32)

    # Y[:, 0] = synthetic regression target
    # Y[:, 1] = binary classification target
    reg = rng.standard_normal(N).astype(np.float32)
    cls = (reg > 0).astype(np.float32)
    Y = np.stack([reg, cls], axis=1)
    return X, Y


@pytest.fixture
def small_tensors():
    """Small (200, 10, 4) tensors simulating EXP_A_PRICE with W=10."""
    return _make_tensors(N=200, W=10, K=4)


@pytest.fixture
def exp_configs():
    """All four experiment configurations as (W, K) pairs."""
    return [
        (60, 4),   # EXP_A
        (60, 5),   # EXP_B
        (60, 19),  # EXP_C
        (60, 24),  # EXP_D
    ]


# ── Flattening Tests ──────────────────────────────────────────────────────────

def test_flatten_output_shape(small_tensors):
    X, _ = small_tensors
    X_flat = flatten_sequences(X)
    assert X_flat.shape == (200, 10 * 4)


def test_flatten_wrong_ndim():
    with pytest.raises(ValueError, match="Expected 3D array"):
        flatten_sequences(np.zeros((100, 240)))


def test_flatten_preserves_values(small_tensors):
    X, _ = small_tensors
    X_flat = flatten_sequences(X)
    # First row of flattened must equal X[0].ravel()
    np.testing.assert_array_equal(X_flat[0], X[0].ravel())


def test_flatten_all_configs(exp_configs):
    for W, K in exp_configs:
        X = np.random.randn(50, W, K)
        X_flat = flatten_sequences(X)
        assert X_flat.shape == (50, W * K), f"Failed for W={W}, K={K}"


# ── Naive Baseline Tests ──────────────────────────────────────────────────────

def test_naive_fit_is_noop(small_tensors):
    X, Y = small_tensors
    model = NaiveBaseline()
    model.fit(X, Y)
    assert model.is_fitted


def test_naive_regression_shape(small_tensors):
    X, Y = small_tensors
    model = NaiveBaseline()
    model.fit(X, Y)
    preds = model.predict_regression(X)
    assert preds.shape == (200,)


def test_naive_classification_valid_values(small_tensors):
    X, Y = small_tensors
    model = NaiveBaseline()
    model.fit(X, Y)
    preds = model.predict_classification(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_naive_momentum_correctness():
    """Verify naive prediction matches X[:, -1, 0] - X[:, -2, 0]."""
    X = np.zeros((5, 10, 4))
    # Set feature 0 at last two steps
    X[:, -2, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X[:, -1, 0] = np.array([2.0, 1.0, 3.0, 6.0, 4.0])
    expected = X[:, -1, 0] - X[:, -2, 0]  # [1, -1, 0, 2, -1]

    model = NaiveBaseline()
    model.fit(X, np.zeros((5, 2)))
    preds = model.predict_regression(X)
    np.testing.assert_array_almost_equal(preds, expected)


def test_naive_refuses_flattened_input():
    X_flat = np.zeros((50, 240))
    model = NaiveBaseline()
    model.fit(X_flat, np.zeros((50, 2)))  # fit is noop
    with pytest.raises(ValueError, match="3D input"):
        model.predict_regression(X_flat)


def test_naive_classification_positive_gets_one():
    X = np.zeros((3, 5, 2))
    X[:, -2, 0] = [1.0, 2.0, 3.0]
    X[:, -1, 0] = [2.0, 1.0, 3.0]
    model = NaiveBaseline()
    model.fit(X, np.zeros((3, 2)))
    cls = model.predict_classification(X)
    assert cls[0] == 1  # momentum > 0
    assert cls[1] == 0  # momentum < 0
    assert cls[2] == 0  # momentum == 0 → 0


# ── Ridge Regression Tests ────────────────────────────────────────────────────

def test_ridge_train_only_and_shape(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    model = RidgeRegressorModel()
    model.fit(X_flat, Y)
    assert model.is_fitted
    preds = model.predict(X_flat)
    assert preds.shape == (200,)
    assert np.all(np.isfinite(preds))


def test_ridge_reproducibility(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    m1 = RidgeRegressorModel()
    m1.fit(X_flat, Y)
    m2 = RidgeRegressorModel()
    m2.fit(X_flat, Y)
    np.testing.assert_array_equal(m1.predict(X_flat), m2.predict(X_flat))


# ── Logistic Regression Tests ─────────────────────────────────────────────────

def test_logistic_train_only_and_valid_output(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    model = LogisticModel()
    model.fit(X_flat, Y)
    assert model.is_fitted
    preds = model.predict(X_flat)
    assert preds.shape == (200,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_logistic_reproducibility(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    m1 = LogisticModel()
    m1.fit(X_flat, Y)
    m2 = LogisticModel()
    m2.fit(X_flat, Y)
    np.testing.assert_array_equal(m1.predict(X_flat), m2.predict(X_flat))


# ── Random Forest Tests ───────────────────────────────────────────────────────

def test_random_forest_train_only(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    model = RandomForestModel()
    model.fit(X_flat, Y)
    assert model.is_fitted

    reg = model.predict_regression(X_flat)
    cls = model.predict_classification(X_flat)

    assert reg.shape == (200,)
    assert cls.shape == (200,)
    assert np.all(np.isfinite(reg))
    assert set(np.unique(cls)).issubset({0, 1})


def test_random_forest_reproducibility(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    hyperparams = {**RandomForestModel.HYPERPARAMS, "n_jobs": 1}

    m1 = RandomForestModel()
    m1.reg_model = RandomForestRegressor(**hyperparams)
    m1.cls_model = RandomForestClassifier(**hyperparams)
    m1.fit(X_flat, Y)

    m2 = RandomForestModel()
    m2.reg_model = RandomForestRegressor(**hyperparams)
    m2.cls_model = RandomForestClassifier(**hyperparams)
    m2.fit(X_flat, Y)

    # RandomForest uses float64 aggregation; identical seeds should produce
    # bit-identical results with n_jobs=1. Use almost_equal for any
    # platform-level float rounding at the ~1e-14 level.
    np.testing.assert_array_almost_equal(
        m1.predict_regression(X_flat),
        m2.predict_regression(X_flat),
        decimal=10,
    )


# ── XGBoost Tests ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed")
def test_xgboost_train_only(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    model = XGBoostModel()
    model.fit(X_flat, Y)
    assert model.is_fitted

    reg = model.predict_regression(X_flat)
    cls = model.predict_classification(X_flat)

    assert reg.shape == (200,)
    assert cls.shape == (200,)
    assert np.all(np.isfinite(reg))
    assert set(np.unique(cls)).issubset({0, 1})


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason="xgboost not installed")
def test_xgboost_reproducibility(small_tensors):
    X, Y = small_tensors
    X_flat = flatten_sequences(X)
    m1 = XGBoostModel()
    m1.fit(X_flat, Y)
    m2 = XGBoostModel()
    m2.fit(X_flat, Y)
    np.testing.assert_array_almost_equal(
        m1.predict_regression(X_flat),
        m2.predict_regression(X_flat),
    )


# ── Metric Tests ──────────────────────────────────────────────────────────────

def test_regression_metrics_correctness():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert pytest.approx(metrics["mae"]) == 0.5
    assert pytest.approx(metrics["rmse"]) == 0.5
    assert "mape" in metrics
    assert "mape_coverage" in metrics


def test_mape_excluded_for_near_zero_returns():
    """Observations where |y_true| < 1e-8 are excluded from MAPE."""
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.5, 1.5, 2.5])
    metrics = compute_regression_metrics(y_true, y_pred)
    # Coverage should be 2/3 (only 2 non-zero true values)
    assert pytest.approx(metrics["mape_coverage"]) == 2 / 3


def test_mape_all_zeros_returns_nan():
    y_true = np.zeros(5)
    y_pred = np.ones(5)
    metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isnan(metrics["mape"])


def test_classification_metrics_correctness():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert pytest.approx(metrics["accuracy"]) == 0.5
    assert "f1_macro" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert isinstance(metrics["confusion_matrix"], list)


def test_classification_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert pytest.approx(metrics["accuracy"]) == 1.0
    assert pytest.approx(metrics["f1_macro"]) == 1.0


# ── All Experiment Configurations ─────────────────────────────────────────────

@pytest.mark.parametrize("W,K", [(60, 4), (60, 5), (60, 19), (60, 24)])
def test_ridge_runs_all_experiments(W, K):
    X, Y = _make_tensors(N=80, W=W, K=K)
    X_flat = flatten_sequences(X)
    assert X_flat.shape == (80, W * K)
    model = RidgeRegressorModel()
    model.fit(X_flat, Y)
    preds = model.predict(X_flat)
    assert preds.shape == (80,)


@pytest.mark.parametrize("W,K", [(60, 4), (60, 5), (60, 19), (60, 24)])
def test_logistic_runs_all_experiments(W, K):
    X, Y = _make_tensors(N=80, W=W, K=K)
    X_flat = flatten_sequences(X)
    model = LogisticModel()
    model.fit(X_flat, Y)
    preds = model.predict(X_flat)
    assert preds.shape == (80,)
    assert set(np.unique(preds)).issubset({0, 1})
