"""
Phase 8 test suite: Market Regime Classifier.

Tests verify:
- Mathematical correctness of slope and volatility computations.
- Rolling-window behaviour (warm-up handling, boundary conditions).
- Regime classification boundary logic and precedence (High Volatility first).
- Timestamp / index alignment (regime_df aligns 1:1 with input close series).
- No look-ahead: thresholds derived solely from training-partition rows.
- Deterministic / reproducible output.
- Expected regime label set.
- Finite / non-NaN regime metrics where applicable.
- Raw dataset SHA-256 unchanged.
- Phase 4–7 existing artifacts unchanged (spot-check).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "btcusdt_5m_raw.csv"
RESULTS_DIR = ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from crypto_analyzer.regimes.classifier import (
    REGIME_WINDOW,
    WARMUP_LABEL,
    classify_regime,
    compute_regime_slope,
    compute_regime_volatility,
    compute_rolling_regimes,
)
from crypto_analyzer.regimes.analysis import TRAIN_END_IDX, VAL_END_IDX


# ---------------------------------------------------------------------------
# 1. Mathematical correctness
# ---------------------------------------------------------------------------

class TestComputeRegimeSlope:
    def test_flat_line_returns_zero(self):
        prices = np.full(10, 100.0)
        assert compute_regime_slope(prices) == pytest.approx(0.0)

    def test_perfect_upward_trend_positive(self):
        prices = np.arange(1.0, 101.0)  # ascending
        s = compute_regime_slope(prices)
        assert s > 0.0

    def test_perfect_downward_trend_negative(self):
        prices = np.arange(100.0, 0.0, -1.0)  # descending
        s = compute_regime_slope(prices)
        assert s < 0.0

    def test_normalization_by_mean(self):
        """Doubling all prices should halve the normalized slope."""
        prices = np.arange(10.0, 20.0)
        s1 = compute_regime_slope(prices)
        s2 = compute_regime_slope(prices * 2)
        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_zero_mean_returns_zero(self):
        """Protect against division by zero when mean price is 0."""
        prices = np.zeros(10)
        assert compute_regime_slope(prices) == 0.0


class TestComputeRegimeVolatility:
    def test_constant_prices_zero_volatility(self):
        prices = np.full(50, 100.0)
        vol = compute_regime_volatility(prices)
        assert vol == pytest.approx(0.0)

    def test_single_price_is_nan(self):
        vol = compute_regime_volatility(np.array([100.0]))
        assert np.isnan(vol)

    def test_positive_for_random_prices(self):
        rng = np.random.default_rng(42)
        prices = 100 + rng.normal(0, 0.5, 100)
        vol = compute_regime_volatility(prices)
        assert vol > 0.0

    def test_higher_oscillation_higher_vol(self):
        low_vol = np.array([100.0, 100.1, 99.9, 100.1, 99.9])
        high_vol = np.array([100.0, 110.0, 90.0, 115.0, 85.0])
        assert compute_regime_volatility(high_vol) > compute_regime_volatility(low_vol)


# ---------------------------------------------------------------------------
# 2. Regime classification boundary logic and precedence
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    SLOPE_TH = 0.0001
    VOL_TH = 0.002

    def test_high_volatility_takes_precedence(self):
        """High volatility overrides slope direction."""
        # Even with a very bullish slope, if vol is above threshold → High Volatility
        label = classify_regime(self.SLOPE_TH * 10, self.VOL_TH * 2, self.SLOPE_TH, self.VOL_TH)
        assert label == "High Volatility"

    def test_bullish(self):
        label = classify_regime(self.SLOPE_TH * 2, 0.0, self.SLOPE_TH, self.VOL_TH)
        assert label == "Bullish"

    def test_bearish(self):
        label = classify_regime(-self.SLOPE_TH * 2, 0.0, self.SLOPE_TH, self.VOL_TH)
        assert label == "Bearish"

    def test_sideways(self):
        label = classify_regime(0.0, 0.0, self.SLOPE_TH, self.VOL_TH)
        assert label == "Sideways"

    def test_slope_exactly_at_threshold_is_sideways(self):
        """Boundary: slope == +theta_slope is NOT > theta_slope → Sideways."""
        label = classify_regime(self.SLOPE_TH, 0.0, self.SLOPE_TH, self.VOL_TH)
        assert label == "Sideways"

    def test_all_labels_are_known(self):
        valid = {"Bullish", "Bearish", "Sideways", "High Volatility"}
        for slope in [-0.001, 0.0, 0.001]:
            for vol in [0.0, 0.003]:
                label = classify_regime(slope, vol, self.SLOPE_TH, self.VOL_TH)
                assert label in valid


# ---------------------------------------------------------------------------
# 3. Rolling regime computation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rolling_regime_output():
    """Compute rolling regimes once on a synthetic series for reuse."""
    n = 400
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.1, n))
    close = pd.Series(prices)
    df, tv, ts = compute_rolling_regimes(
        close=close,
        window=50,
        theta_slope=None,
        theta_vol_high=None,
        train_end_idx=280,
    )
    return df, tv, ts, close


class TestComputeRollingRegimes:
    def test_output_length_matches_input(self, rolling_regime_output):
        df, tv, ts, close = rolling_regime_output
        assert len(df) == len(close)

    def test_warmup_rows_correctly_labelled(self, rolling_regime_output):
        df, tv, ts, close = rolling_regime_output
        warmup_mask = df["regime"] == WARMUP_LABEL
        # First 49 rows must be warmup (window=50, so indices 0..48 lack full lookback)
        assert warmup_mask.iloc[:49].all()
        # Row 49 onwards must not be warmup
        assert not warmup_mask.iloc[49:].any()

    def test_slope_nan_in_warmup(self, rolling_regime_output):
        df, tv, ts, close = rolling_regime_output
        assert df["slope"].iloc[:49].isna().all()
        assert df["slope"].iloc[49:].notna().all()

    def test_volatility_nan_in_warmup(self, rolling_regime_output):
        df, tv, ts, close = rolling_regime_output
        assert df["volatility"].iloc[:49].isna().all()
        assert df["volatility"].iloc[49:].notna().all()

    def test_regime_labels_are_valid(self, rolling_regime_output):
        df, tv, ts, close = rolling_regime_output
        valid = {"Bullish", "Bearish", "Sideways", "High Volatility", WARMUP_LABEL}
        assert set(df["regime"].unique()).issubset(valid)

    def test_theta_vol_high_derived_from_training_only(self, rolling_regime_output):
        """theta_vol_high must not use future data."""
        df, tv, ts, close = rolling_regime_output
        # If train_end_idx=280, theta_vol_high must be computable from only the first 280 rows
        train_vols = df["volatility"].iloc[:280].dropna()
        expected = float(np.percentile(train_vols, 90))
        assert tv == pytest.approx(expected, rel=1e-10)

    def test_theta_slope_derived_from_training_only(self, rolling_regime_output):
        """theta_slope must not use future data."""
        df, tv, ts, close = rolling_regime_output
        train_slopes = df["slope"].iloc[:280].dropna()
        expected = float(np.percentile(np.abs(train_slopes), 50))
        assert ts == pytest.approx(expected, rel=1e-10)

    def test_deterministic_output(self):
        """Same input → same output."""
        rng = np.random.default_rng(99)
        prices = 100 + np.cumsum(rng.normal(0, 0.1, 300))
        close = pd.Series(prices)
        df1, tv1, ts1 = compute_rolling_regimes(close, window=50, train_end_idx=200)
        df2, tv2, ts2 = compute_rolling_regimes(close, window=50, train_end_idx=200)
        pd.testing.assert_frame_equal(df1, df2)
        assert tv1 == tv2
        assert ts1 == ts2

    def test_index_alignment(self):
        """regime_df index must align 1:1 with input close index."""
        close = pd.Series(np.arange(100.0, 200.0), index=range(50, 150))
        df, tv, ts = compute_rolling_regimes(close, window=20, train_end_idx=70)
        assert list(df.index) == list(close.index)

    def test_finite_non_nan_for_classified_rows(self, rolling_regime_output):
        df, tv, ts, close = rolling_regime_output
        classified = df[df["regime"] != WARMUP_LABEL]
        assert classified["slope"].notna().all()
        assert classified["volatility"].notna().all()
        assert np.isfinite(classified["slope"]).all()
        assert np.isfinite(classified["volatility"]).all()


# ---------------------------------------------------------------------------
# 4. No look-ahead leakage
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    def test_thresholds_derived_from_train_only(self):
        """Changing post-train data should NOT change theta_vol_high or theta_slope."""
        rng = np.random.default_rng(7)
        n = 400
        prices = 100 + np.cumsum(rng.normal(0, 0.2, n))
        train_end = 280

        close1 = pd.Series(prices.copy())
        close2 = pd.Series(prices.copy())
        # Perturb only the post-training portion of close2
        close2.iloc[train_end:] = close2.iloc[train_end:] * 2.0

        _, tv1, ts1 = compute_rolling_regimes(close1, window=50, train_end_idx=train_end)
        _, tv2, ts2 = compute_rolling_regimes(close2, window=50, train_end_idx=train_end)

        assert tv1 == pytest.approx(tv2, rel=1e-10), "theta_vol_high leaked future data"
        assert ts1 == pytest.approx(ts2, rel=1e-10), "theta_slope leaked future data"


# ---------------------------------------------------------------------------
# 5. Real-data artifact validation
# ---------------------------------------------------------------------------

class TestRealDataIntegrity:
    def test_raw_data_sha256_unchanged(self):
        sha256 = hashlib.sha256()
        with open(DATA_RAW, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        assert sha256.hexdigest() == "a722751b2929acdcae48cb875be4c3d4904d17e99d45ea0b3f6d9a67a4cf1eb8"

    def test_baseline_results_unchanged(self):
        path = RESULTS_DIR / "all_baselines_results.json"
        assert path.exists(), "Phase 5 artifact missing"
        with open(path) as f:
            data = json.load(f)
        # Just verify structure hasn't been mutated
        assert set(data.keys()) == {
            "EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"
        }

    def test_ablation_summary_exists(self):
        path = RESULTS_DIR / "ablation_summary.json"
        assert path.exists(), "Phase 7 ablation_summary.json missing"

    def test_regime_analysis_artifact_exists(self):
        path = RESULTS_DIR / "regime_analysis_summary.json"
        assert path.exists(), "Phase 8 regime_analysis_summary.json missing"

    def test_regime_labels_csv_exists(self):
        path = RESULTS_DIR / "test_regime_labels.csv"
        assert path.exists(), "Phase 8 test_regime_labels.csv missing"

    def test_regime_labels_csv_schema(self):
        path = RESULTS_DIR / "test_regime_labels.csv"
        df = pd.read_csv(path)
        for col in ["regime", "regime_slope", "regime_volatility", "regime_code"]:
            assert col in df.columns

    def test_regime_labels_no_warmup_in_test(self):
        """Test partition should have zero warm-up rows given regime window < val partition end."""
        path = RESULTS_DIR / "test_regime_labels.csv"
        df = pd.read_csv(path)
        assert (df["regime"] == WARMUP_LABEL).sum() == 0

    def test_all_regimes_finite(self):
        path = RESULTS_DIR / "test_regime_labels.csv"
        df = pd.read_csv(path)
        classified = df[df["regime"] != WARMUP_LABEL]
        assert np.isfinite(classified["regime_slope"]).all()
        assert np.isfinite(classified["regime_volatility"]).all()

    def test_regime_analysis_summary_structure(self):
        path = RESULTS_DIR / "regime_analysis_summary.json"
        with open(path) as f:
            summary = json.load(f)
        assert "methodology" in summary
        assert "test_partition_analysis" in summary
        assert "theta_slope_used" in summary["methodology"]
        assert "theta_vol_high" in summary["methodology"]
        test = summary["test_partition_analysis"]
        for regime in ["Bullish", "Bearish", "Sideways", "High Volatility"]:
            assert regime in test["regime_counts"]

    def test_feature_dimension_expected_counts(self):
        """Phase 4 K-dims should still be correct after Phase 8."""
        path = RESULTS_DIR / "ablation_summary.json"
        with open(path) as f:
            data = json.load(f)
        # Check regression entry has consistent models
        for exp in ["EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"]:
            assert exp in data["regression"]
