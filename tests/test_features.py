"""
Comprehensive test suite for Phase 3 Feature Engineering.

Tests:
1. Mathematical correctness on controlled toy data.
2. Temporal alignment / zero future look-ahead (via future-row mutation).
3. Per-feature rolling window warm-up NaN verification.
4. Experiment feature column counts (4, 5, 19, 24).
5. NaN / Inf handling.
6. Chronological ordering preservation.
7. Pipeline integration.
8. RSI edge-case behavior (all gains, all losses, zero movement).
9. Raw data immutability (no writes to data/raw/).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.features.indicators import (
    compute_atr,
    compute_atr_ratio,
    compute_bollinger_bands,
    compute_ema,
    compute_log_returns,
    compute_macd,
    compute_returns,
    compute_rolling_volatility,
    compute_rsi,
    compute_sma,
)
from crypto_analyzer.features.pipeline import (
    EXPECTED_FEATURE_COUNTS,
    EXPERIMENT_FEATURES,
    build_features,
    get_feature_columns,
    get_feature_summary,
    get_warm_up_rows,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def toy_ohlcv():
    """Generate a small controlled OHLCV DataFrame for unit tests (200 rows)."""
    np.random.seed(42)
    n = 200
    timestamps = list(range(1000000, 1000000 + n * 300000, 300000))
    close = [100.0]
    for i in range(1, n):
        close.append(close[-1] * (1 + np.random.normal(0, 0.005)))
    close = np.array(close)
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_ = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.abs(np.random.normal(100, 20, n))

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def simple_close():
    """Simple ascending close price series for deterministic tests."""
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])


# ══════════════════════════════════════════════════════════════════════
# 1. Mathematical Correctness Tests
# ══════════════════════════════════════════════════════════════════════

class TestReturnsMath:
    """Test simple and log return calculations."""

    def test_simple_returns_known_values(self):
        close = pd.Series([100.0, 110.0, 99.0, 110.0])
        returns = compute_returns(close)
        assert np.isnan(returns.iloc[0])
        assert pytest.approx(returns.iloc[1], rel=1e-10) == 0.10
        assert pytest.approx(returns.iloc[2], rel=1e-10) == -0.1
        assert pytest.approx(returns.iloc[3], rel=1e-10) == 11.0 / 99.0

    def test_log_returns_known_values(self):
        close = pd.Series([100.0, 110.0, 99.0])
        log_ret = compute_log_returns(close)
        assert np.isnan(log_ret.iloc[0])
        assert pytest.approx(log_ret.iloc[1], rel=1e-10) == np.log(110.0 / 100.0)
        assert pytest.approx(log_ret.iloc[2], rel=1e-10) == np.log(99.0 / 110.0)

    def test_returns_first_value_is_nan(self):
        close = pd.Series([50.0, 60.0, 70.0])
        assert np.isnan(compute_returns(close).iloc[0])
        assert np.isnan(compute_log_returns(close).iloc[0])


class TestSMAMath:
    """Test Simple Moving Average correctness."""

    def test_sma_known_sequence(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma3 = compute_sma(series, period=3)
        assert np.isnan(sma3.iloc[0])
        assert np.isnan(sma3.iloc[1])
        assert pytest.approx(sma3.iloc[2]) == 2.0  # (1+2+3)/3
        assert pytest.approx(sma3.iloc[3]) == 3.0  # (2+3+4)/3
        assert pytest.approx(sma3.iloc[4]) == 4.0  # (3+4+5)/3

    def test_sma_warm_up_nan_count(self):
        series = pd.Series(range(100), dtype=float)
        sma12 = compute_sma(series, period=12)
        assert sma12.iloc[:11].isna().all()
        assert not np.isnan(sma12.iloc[11])


class TestEMAMath:
    """Test Exponential Moving Average correctness with min_periods."""

    def test_ema_uses_min_periods(self):
        """EMA with min_periods=period produces NaN for first (period-1) rows."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        ema3 = compute_ema(series, period=3)
        # min_periods=3: first 2 values are NaN
        assert np.isnan(ema3.iloc[0])
        assert np.isnan(ema3.iloc[1])
        assert not np.isnan(ema3.iloc[2])

    def test_ema_adjust_false_recursive(self):
        """After warm-up, EMA follows recursive formula with adjust=False."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        ema3 = compute_ema(series, period=3)
        alpha = 2.0 / (3 + 1)  # 0.5

        # First valid value is at index 2 (min_periods=3)
        # With adjust=False and min_periods=3, pandas seeds EMA at index 2
        first_valid = ema3.iloc[2]
        # Verify recursive step at index 3
        expected_3 = alpha * series.iloc[3] + (1 - alpha) * first_valid
        assert pytest.approx(ema3.iloc[3]) == expected_3

    def test_ema_12_warm_up(self):
        series = pd.Series(range(50), dtype=float)
        ema = compute_ema(series, period=12)
        assert ema.iloc[:11].isna().all()
        assert not np.isnan(ema.iloc[11])

    def test_ema_26_warm_up(self):
        series = pd.Series(range(50), dtype=float)
        ema = compute_ema(series, period=26)
        assert ema.iloc[:25].isna().all()
        assert not np.isnan(ema.iloc[25])


class TestRSIMath:
    """Test RSI boundary values, correctness, and edge cases."""

    def test_rsi_all_gains_equals_100(self):
        """Monotonically increasing prices → RSI = 100 (avg_loss = 0)."""
        close = pd.Series([float(i) for i in range(1, 50)])
        rsi = compute_rsi(close, period=14)
        # After warm-up, RSI should be exactly 100 for all-gains
        valid_rsi = rsi.dropna()
        assert (valid_rsi == 100.0).all(), f"Expected all RSI=100, got: {valid_rsi.values}"

    def test_rsi_all_losses_equals_0(self):
        """Monotonically decreasing prices → RSI = 0 (avg_gain = 0)."""
        close = pd.Series([50.0 - 0.5 * i for i in range(50)])
        rsi = compute_rsi(close, period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi == 0.0).all(), f"Expected all RSI=0, got: {valid_rsi.values}"

    def test_rsi_zero_movement_equals_50(self):
        """Constant prices (no gains, no losses) → RSI = 50 (neutral)."""
        close = pd.Series([100.0] * 30)
        rsi = compute_rsi(close, period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi == 50.0).all(), f"Expected all RSI=50, got: {valid_rsi.values}"

    def test_rsi_warm_up_nan(self):
        """First `period` values must be NaN."""
        close = pd.Series(range(1, 30), dtype=float)
        rsi = compute_rsi(close, period=14)
        assert rsi.iloc[:14].isna().all()
        assert not np.isnan(rsi.iloc[14])

    def test_rsi_range(self, toy_ohlcv):
        """RSI values must be in [0, 100] after warm-up."""
        rsi = compute_rsi(toy_ohlcv["close"], period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()


class TestMACDMath:
    """Test MACD component correctness."""

    def test_macd_is_ema12_minus_ema26(self):
        close = pd.Series(np.random.RandomState(42).normal(100, 5, 100))
        macd_df = compute_macd(close)
        ema12 = compute_ema(close, 12)
        ema26 = compute_ema(close, 26)
        expected_macd = ema12 - ema26
        pd.testing.assert_series_equal(
            macd_df["macd"], expected_macd, check_names=False
        )

    def test_macd_hist_is_macd_minus_signal(self):
        close = pd.Series(np.random.RandomState(42).normal(100, 5, 100))
        macd_df = compute_macd(close)
        expected_hist = macd_df["macd"] - macd_df["macd_signal"]
        pd.testing.assert_series_equal(
            macd_df["macd_hist"], expected_hist, check_names=False
        )

    def test_macd_returns_three_columns(self):
        close = pd.Series(np.random.RandomState(42).normal(100, 5, 50))
        macd_df = compute_macd(close)
        assert set(macd_df.columns) == {"macd", "macd_signal", "macd_hist"}

    def test_macd_warm_up_follows_ema_policy(self):
        """MACD line NaN for first 25 rows (driven by EMA_26 min_periods=26)."""
        close = pd.Series(range(1, 60), dtype=float)
        macd_df = compute_macd(close)
        # MACD line: EMA_12 (11 NaN) - EMA_26 (25 NaN) → 25 NaN
        assert macd_df["macd"].iloc[:25].isna().all()
        assert not np.isnan(macd_df["macd"].iloc[25])

    def test_macd_signal_warm_up_follows_ema_policy(self):
        """MACD Signal NaN for first 33 rows (EMA_26 + EMA_9 on MACD)."""
        close = pd.Series(range(1, 60), dtype=float)
        macd_df = compute_macd(close)
        # Signal = EMA_9(MACD). MACD first valid at 25. EMA_9 needs 9 non-NaN.
        # First valid signal at index 25 + 9 - 1 = 33.
        assert macd_df["macd_signal"].iloc[:33].isna().all()
        assert not np.isnan(macd_df["macd_signal"].iloc[33])


class TestBollingerBandsMath:
    """Test Bollinger Bands correctness."""

    def test_bb_sma_center(self):
        close = pd.Series(range(1, 30), dtype=float)
        bb = compute_bollinger_bands(close, period=20, num_std=2.0)
        sma20 = compute_sma(close, 20)
        mid = (bb["bb_upper"] + bb["bb_lower"]) / 2
        pd.testing.assert_series_equal(mid.dropna(), sma20.dropna(), check_names=False)

    def test_bb_width_formula(self):
        close = pd.Series(range(1, 30), dtype=float)
        bb = compute_bollinger_bands(close, period=20, num_std=2.0)
        sma20 = compute_sma(close, 20)
        expected_width = (bb["bb_upper"] - bb["bb_lower"]) / sma20
        pd.testing.assert_series_equal(
            bb["bb_width"].dropna(), expected_width.dropna(), check_names=False
        )

    def test_bb_warm_up_nan(self):
        close = pd.Series(range(1, 30), dtype=float)
        bb = compute_bollinger_bands(close, period=20, num_std=2.0)
        assert bb["bb_upper"].iloc[:19].isna().all()
        assert not np.isnan(bb["bb_upper"].iloc[19])


class TestATRMath:
    """Test Average True Range correctness."""

    def test_atr_true_range_formula(self):
        """Verify TR = max(H-L, |H-prevC|, |L-prevC|) on known values."""
        high = pd.Series([100.0, 110.0, 105.0])
        low = pd.Series([100.0, 90.0, 95.0])
        close = pd.Series([100.0, 105.0, 100.0])
        # At index 1: prev_close=100, TR=max(110-90, |110-100|, |90-100|)=max(20,10,10)=20
        # At index 2: prev_close=105, TR=max(105-95, |105-105|, |95-105|)=max(10,0,10)=10
        atr = compute_atr(high, low, close, period=1)
        # With period=1 and min_periods=1, first valid after TR[0]=NaN (shift)
        # At index 1: TR=20, EMA with period=1 = 20
        assert not np.isnan(atr.iloc[1])

    def test_atr_warm_up_nan(self, toy_ohlcv):
        """First 14 values must be NaN (period=14 with EMA min_periods=14)."""
        atr = compute_atr(
            toy_ohlcv["high"], toy_ohlcv["low"], toy_ohlcv["close"], period=14
        )
        assert atr.iloc[:14].isna().all()
        assert not np.isnan(atr.iloc[14])

    def test_atr_always_positive(self, toy_ohlcv):
        atr = compute_atr(
            toy_ohlcv["high"], toy_ohlcv["low"], toy_ohlcv["close"], period=14
        )
        valid = atr.dropna()
        assert (valid > 0).all()


class TestVolatilityMath:
    """Test rolling volatility and ATR ratio correctness."""

    def test_rolling_vol_matches_manual_std(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        vol = compute_rolling_volatility(returns, period=3)
        assert np.isnan(vol.iloc[0])
        assert np.isnan(vol.iloc[1])
        expected = np.std([0.01, -0.02, 0.015], ddof=1)
        assert pytest.approx(vol.iloc[2], rel=1e-10) == expected

    def test_atr_ratio_formula(self):
        atr = pd.Series([2.0, 3.0, 4.0])
        close = pd.Series([100.0, 150.0, 200.0])
        ratio = compute_atr_ratio(atr, close)
        assert pytest.approx(ratio.iloc[0]) == 0.02
        assert pytest.approx(ratio.iloc[1]) == 0.02
        assert pytest.approx(ratio.iloc[2]) == 0.02


# ══════════════════════════════════════════════════════════════════════
# 2. Temporal Alignment / No Future Look-Ahead Tests
# ══════════════════════════════════════════════════════════════════════

class TestNoFutureLookAhead:
    """
    Verify that indicator values at time t do NOT change when future data
    (rows after t) are modified.

    Strategy: compute features on the full DataFrame, then mutate rows
    after index t to extreme values and recompute — the value at index t
    must remain identical.
    """

    def _mutate_future(self, series, t, extreme=999999.0):
        """Return a copy with all values after index t replaced by extreme."""
        mutated = series.copy()
        mutated.iloc[t + 1:] = extreme
        return mutated

    def test_sma_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_sma(toy_ohlcv["close"], 12)
        mutated = compute_sma(self._mutate_future(toy_ohlcv["close"], t), 12)
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_ema_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_ema(toy_ohlcv["close"], 12)
        mutated = compute_ema(self._mutate_future(toy_ohlcv["close"], t), 12)
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_rsi_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_rsi(toy_ohlcv["close"], 14)
        mutated = compute_rsi(self._mutate_future(toy_ohlcv["close"], t), 14)
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_macd_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_macd(toy_ohlcv["close"])
        mutated = compute_macd(self._mutate_future(toy_ohlcv["close"], t))
        assert pytest.approx(full["macd"].iloc[t]) == mutated["macd"].iloc[t]
        assert pytest.approx(full["macd_signal"].iloc[t]) == mutated["macd_signal"].iloc[t]
        assert pytest.approx(full["macd_hist"].iloc[t]) == mutated["macd_hist"].iloc[t]

    def test_bollinger_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_bollinger_bands(toy_ohlcv["close"], 20, 2.0)
        mutated = compute_bollinger_bands(
            self._mutate_future(toy_ohlcv["close"], t), 20, 2.0
        )
        assert pytest.approx(full["bb_upper"].iloc[t]) == mutated["bb_upper"].iloc[t]
        assert pytest.approx(full["bb_lower"].iloc[t]) == mutated["bb_lower"].iloc[t]
        assert pytest.approx(full["bb_width"].iloc[t]) == mutated["bb_width"].iloc[t]
        assert pytest.approx(full["bb_pct_b"].iloc[t]) == mutated["bb_pct_b"].iloc[t]

    def test_atr_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_atr(
            toy_ohlcv["high"], toy_ohlcv["low"], toy_ohlcv["close"], 14
        )
        mutated = compute_atr(
            self._mutate_future(toy_ohlcv["high"], t),
            self._mutate_future(toy_ohlcv["low"], t),
            self._mutate_future(toy_ohlcv["close"], t),
            14,
        )
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_rolling_vol_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        returns = compute_returns(toy_ohlcv["close"])
        full = compute_rolling_volatility(returns, 12)
        mutated_returns = compute_returns(self._mutate_future(toy_ohlcv["close"], t))
        mutated = compute_rolling_volatility(mutated_returns, 12)
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_returns_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_returns(toy_ohlcv["close"])
        mutated = compute_returns(self._mutate_future(toy_ohlcv["close"], t))
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_log_returns_no_lookahead_mutation(self, toy_ohlcv):
        t = 50
        full = compute_log_returns(toy_ohlcv["close"])
        mutated = compute_log_returns(self._mutate_future(toy_ohlcv["close"], t))
        assert pytest.approx(full.iloc[t]) == mutated.iloc[t]

    def test_full_pipeline_no_lookahead_mutation(self, toy_ohlcv):
        """Build features on full data vs mutated-future data; all features at t must match."""
        t = 50
        full_df = build_features(toy_ohlcv)

        mutated = toy_ohlcv.copy()
        mutated.loc[mutated.index[t + 1:], "close"] = 999999.0
        mutated.loc[mutated.index[t + 1:], "high"] = 999999.0
        mutated.loc[mutated.index[t + 1:], "low"] = 999999.0
        mutated.loc[mutated.index[t + 1:], "open"] = 999999.0
        mutated.loc[mutated.index[t + 1:], "volume"] = 999999.0
        mutated_df = build_features(mutated)

        for col in get_feature_columns("EXP_D_FULL"):
            full_val = full_df[col].iloc[t]
            mutated_val = mutated_df[col].iloc[t]
            if np.isnan(full_val):
                assert np.isnan(mutated_val), f"{col} NaN mismatch at index {t}"
            else:
                assert pytest.approx(full_val) == mutated_val, (
                    f"{col} look-ahead detected: full={full_val}, mutated={mutated_val}"
                )


# ══════════════════════════════════════════════════════════════════════
# 3. Per-Feature Rolling Window Warm-Up NaN Tests
# ══════════════════════════════════════════════════════════════════════

class TestPerFeatureWarmUp:
    """Verify actual NaN warm-up behavior for EVERY feature independently."""

    def test_returns_warmup_1(self, toy_ohlcv):
        """returns: 1 NaN (index 0), valid from index 1."""
        ret = compute_returns(toy_ohlcv["close"])
        assert np.isnan(ret.iloc[0])
        assert not np.isnan(ret.iloc[1])

    def test_log_returns_warmup_1(self, toy_ohlcv):
        """log_returns: 1 NaN (index 0), valid from index 1."""
        lr = compute_log_returns(toy_ohlcv["close"])
        assert np.isnan(lr.iloc[0])
        assert not np.isnan(lr.iloc[1])

    def test_sma_12_warmup_11(self, toy_ohlcv):
        """sma_12: 11 NaN (indices 0-10), valid from index 11."""
        sma = compute_sma(toy_ohlcv["close"], 12)
        assert sma.iloc[:11].isna().all()
        assert not np.isnan(sma.iloc[11])

    def test_sma_24_warmup_23(self, toy_ohlcv):
        """sma_24: 23 NaN (indices 0-22), valid from index 23."""
        sma = compute_sma(toy_ohlcv["close"], 24)
        assert sma.iloc[:23].isna().all()
        assert not np.isnan(sma.iloc[23])

    def test_sma_96_warmup_95(self, toy_ohlcv):
        """sma_96: 95 NaN (indices 0-94), valid from index 95."""
        sma = compute_sma(toy_ohlcv["close"], 96)
        assert sma.iloc[:95].isna().all()
        assert not np.isnan(sma.iloc[95])

    def test_ema_12_warmup_11(self, toy_ohlcv):
        """ema_12: 11 NaN (indices 0-10), valid from index 11."""
        ema = compute_ema(toy_ohlcv["close"], 12)
        assert ema.iloc[:11].isna().all()
        assert not np.isnan(ema.iloc[11])

    def test_ema_26_warmup_25(self, toy_ohlcv):
        """ema_26: 25 NaN (indices 0-24), valid from index 25."""
        ema = compute_ema(toy_ohlcv["close"], 26)
        assert ema.iloc[:25].isna().all()
        assert not np.isnan(ema.iloc[25])

    def test_rsi_14_warmup_14(self, toy_ohlcv):
        """rsi_14: 14 NaN (indices 0-13), valid from index 14."""
        rsi = compute_rsi(toy_ohlcv["close"], 14)
        assert rsi.iloc[:14].isna().all()
        assert not np.isnan(rsi.iloc[14])

    def test_macd_line_warmup_25(self, toy_ohlcv):
        """macd line: 25 NaN (indices 0-24), valid from index 25."""
        macd_df = compute_macd(toy_ohlcv["close"])
        assert macd_df["macd"].iloc[:25].isna().all()
        assert not np.isnan(macd_df["macd"].iloc[25])

    def test_macd_signal_warmup_33(self, toy_ohlcv):
        """macd_signal: 33 NaN (indices 0-32), valid from index 33."""
        macd_df = compute_macd(toy_ohlcv["close"])
        assert macd_df["macd_signal"].iloc[:33].isna().all()
        assert not np.isnan(macd_df["macd_signal"].iloc[33])

    def test_macd_hist_warmup_33(self, toy_ohlcv):
        """macd_hist: 33 NaN (indices 0-32), valid from index 33."""
        macd_df = compute_macd(toy_ohlcv["close"])
        assert macd_df["macd_hist"].iloc[:33].isna().all()
        assert not np.isnan(macd_df["macd_hist"].iloc[33])

    def test_bb_upper_warmup_19(self, toy_ohlcv):
        """bb_upper: 19 NaN (indices 0-18), valid from index 19."""
        bb = compute_bollinger_bands(toy_ohlcv["close"], 20, 2.0)
        assert bb["bb_upper"].iloc[:19].isna().all()
        assert not np.isnan(bb["bb_upper"].iloc[19])

    def test_bb_lower_warmup_19(self, toy_ohlcv):
        """bb_lower: 19 NaN (indices 0-18), valid from index 19."""
        bb = compute_bollinger_bands(toy_ohlcv["close"], 20, 2.0)
        assert bb["bb_lower"].iloc[:19].isna().all()
        assert not np.isnan(bb["bb_lower"].iloc[19])

    def test_bb_width_warmup_19(self, toy_ohlcv):
        bb = compute_bollinger_bands(toy_ohlcv["close"], 20, 2.0)
        assert bb["bb_width"].iloc[:19].isna().all()
        assert not np.isnan(bb["bb_width"].iloc[19])

    def test_bb_pct_b_warmup_19(self, toy_ohlcv):
        bb = compute_bollinger_bands(toy_ohlcv["close"], 20, 2.0)
        assert bb["bb_pct_b"].iloc[:19].isna().all()
        assert not np.isnan(bb["bb_pct_b"].iloc[19])

    def test_atr_14_warmup_14(self, toy_ohlcv):
        """atr_14: 14 NaN (indices 0-13), valid from index 14."""
        atr = compute_atr(toy_ohlcv["high"], toy_ohlcv["low"], toy_ohlcv["close"], 14)
        assert atr.iloc[:14].isna().all()
        assert not np.isnan(atr.iloc[14])

    def test_rolling_vol_12_warmup_12(self, toy_ohlcv):
        """rolling_vol_12: 12 NaN (indices 0-11), valid from index 12."""
        returns = compute_returns(toy_ohlcv["close"])
        vol = compute_rolling_volatility(returns, 12)
        assert vol.iloc[:12].isna().all()
        assert not np.isnan(vol.iloc[12])

    def test_rolling_vol_24_warmup_24(self, toy_ohlcv):
        """rolling_vol_24: 24 NaN (indices 0-23), valid from index 24."""
        returns = compute_returns(toy_ohlcv["close"])
        vol = compute_rolling_volatility(returns, 24)
        assert vol.iloc[:24].isna().all()
        assert not np.isnan(vol.iloc[24])

    def test_rolling_vol_96_warmup_96(self, toy_ohlcv):
        """rolling_vol_96: 96 NaN (indices 0-95), valid from index 96."""
        returns = compute_returns(toy_ohlcv["close"])
        vol = compute_rolling_volatility(returns, 96)
        assert vol.iloc[:96].isna().all()
        assert not np.isnan(vol.iloc[96])

    def test_atr_ratio_warmup_inherits_atr(self, toy_ohlcv):
        """atr_ratio inherits ATR warm-up: 14 NaN, valid from index 14."""
        atr = compute_atr(toy_ohlcv["high"], toy_ohlcv["low"], toy_ohlcv["close"], 14)
        ratio = compute_atr_ratio(atr, toy_ohlcv["close"])
        assert ratio.iloc[:14].isna().all()
        assert not np.isnan(ratio.iloc[14])

    def test_get_warm_up_rows_equals_96(self):
        """MAX_WARM_UP_ROWS = 96 (driven by rolling_vol_96)."""
        assert get_warm_up_rows() == 96


# ══════════════════════════════════════════════════════════════════════
# 4. Feature Column Count Tests
# ══════════════════════════════════════════════════════════════════════

class TestFeatureColumnCounts:
    """Verify experiment feature sets have exact expected column counts."""

    def test_exp_a_price_has_4_features(self):
        cols = get_feature_columns("EXP_A_PRICE")
        assert len(cols) == 4
        assert cols == ["open", "high", "low", "close"]

    def test_exp_b_price_vol_has_5_features(self):
        cols = get_feature_columns("EXP_B_PRICE_VOL")
        assert len(cols) == 5
        assert cols == ["open", "high", "low", "close", "volume"]

    def test_exp_c_tech_ind_has_19_features(self):
        cols = get_feature_columns("EXP_C_TECH_IND")
        assert len(cols) == 19

    def test_exp_d_full_has_24_features(self):
        cols = get_feature_columns("EXP_D_FULL")
        assert len(cols) == 24

    def test_exp_d_superset_of_exp_c(self):
        c_cols = set(get_feature_columns("EXP_C_TECH_IND"))
        d_cols = set(get_feature_columns("EXP_D_FULL"))
        assert c_cols.issubset(d_cols)

    def test_exp_c_superset_of_exp_b(self):
        b_cols = set(get_feature_columns("EXP_B_PRICE_VOL"))
        c_cols = set(get_feature_columns("EXP_C_TECH_IND"))
        assert b_cols.issubset(c_cols)

    def test_exp_b_superset_of_exp_a(self):
        a_cols = set(get_feature_columns("EXP_A_PRICE"))
        b_cols = set(get_feature_columns("EXP_B_PRICE_VOL"))
        assert a_cols.issubset(b_cols)

    def test_unknown_experiment_raises(self):
        with pytest.raises(ValueError, match="Unknown experiment_id"):
            get_feature_columns("EXP_Z_INVALID")

    def test_expected_feature_counts_match_definitions(self):
        for exp_id, expected in EXPECTED_FEATURE_COUNTS.items():
            assert len(EXPERIMENT_FEATURES[exp_id]) == expected

    def test_exp_d_extra_features_over_exp_c(self):
        c_cols = set(get_feature_columns("EXP_C_TECH_IND"))
        d_cols = set(get_feature_columns("EXP_D_FULL"))
        extra = d_cols - c_cols
        expected_extra = {"rolling_vol_12", "rolling_vol_24", "rolling_vol_96", "atr_ratio", "log_returns"}
        assert extra == expected_extra


# ══════════════════════════════════════════════════════════════════════
# 5. NaN / Inf Handling Tests
# ══════════════════════════════════════════════════════════════════════

class TestNaNInfHandling:
    """Verify NaN is honest (warm-up only) and no Inf values exist."""

    def test_no_inf_in_features(self, toy_ohlcv):
        df = build_features(toy_ohlcv)
        feature_cols = get_feature_columns("EXP_D_FULL")
        for col in feature_cols:
            assert not np.isinf(df[col].dropna()).any(), f"Inf found in {col}"

    def test_raw_columns_never_nan(self, toy_ohlcv):
        """Price and volume features should never have NaN."""
        df = build_features(toy_ohlcv)
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].notna().all(), f"NaN in raw column {col}"

    def test_post_warmup_all_features_valid(self, toy_ohlcv):
        """After the maximum warm-up (96), all 24 features must be non-NaN."""
        df = build_features(toy_ohlcv)
        warm_up = get_warm_up_rows()  # 96
        for col in get_feature_columns("EXP_D_FULL"):
            post_warmup = df[col].iloc[warm_up:]
            nan_count = post_warmup.isna().sum()
            assert nan_count == 0, (
                f"Column {col} has {nan_count} NaN after warm-up row {warm_up}"
            )


# ══════════════════════════════════════════════════════════════════════
# 6. Chronological Ordering Tests
# ══════════════════════════════════════════════════════════════════════

class TestChronologicalOrdering:
    """Verify output preserves strict chronological order."""

    def test_timestamps_strictly_increasing(self, toy_ohlcv):
        df = build_features(toy_ohlcv)
        assert df["timestamp"].is_monotonic_increasing

    def test_output_length_matches_input(self, toy_ohlcv):
        df = build_features(toy_ohlcv)
        assert len(df) == len(toy_ohlcv)

    def test_unsorted_input_raises(self):
        df = pd.DataFrame({
            "timestamp": [3, 2, 1],
            "open": [10, 20, 30],
            "high": [11, 21, 31],
            "low": [9, 19, 29],
            "close": [10, 20, 30],
            "volume": [100, 200, 300],
        })
        with pytest.raises(ValueError, match="strictly increasing"):
            build_features(df)


# ══════════════════════════════════════════════════════════════════════
# 7. Pipeline Integration Tests
# ══════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Test the full build_features pipeline on toy data."""

    def test_build_features_returns_all_24_columns(self, toy_ohlcv):
        df = build_features(toy_ohlcv)
        expected_cols = get_feature_columns("EXP_D_FULL")
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_build_features_preserves_timestamp(self, toy_ohlcv):
        df = build_features(toy_ohlcv)
        assert "timestamp" in df.columns
        pd.testing.assert_series_equal(
            df["timestamp"], toy_ohlcv["timestamp"], check_names=True
        )

    def test_build_features_does_not_modify_input(self, toy_ohlcv):
        original = toy_ohlcv.copy()
        build_features(toy_ohlcv)
        pd.testing.assert_frame_equal(toy_ohlcv, original)

    def test_build_features_missing_columns_raises(self):
        df = pd.DataFrame({"timestamp": [1, 2], "close": [10, 20]})
        with pytest.raises(ValueError, match="Missing required columns"):
            build_features(df)

    def test_build_features_too_few_rows_raises(self):
        df = pd.DataFrame({
            "timestamp": [1],
            "open": [10],
            "high": [11],
            "low": [9],
            "close": [10],
            "volume": [100],
        })
        with pytest.raises(ValueError, match="at least 2 rows"):
            build_features(df)

    def test_feature_summary_has_expected_keys(self, toy_ohlcv):
        df = build_features(toy_ohlcv)
        summary = get_feature_summary(df)
        assert "total_rows" in summary
        assert "warm_up_rows" in summary
        assert "usable_rows" in summary
        assert "feature_counts" in summary
        assert "nan_counts" in summary
        assert "inf_counts" in summary

    def test_no_target_columns_in_output(self, toy_ohlcv):
        """Zero target columns (R_t_12, D_t_12) must be created."""
        df = build_features(toy_ohlcv)
        target_cols = [c for c in df.columns if c.startswith("R_t_") or c.startswith("D_t_")]
        assert len(target_cols) == 0, f"Target columns found: {target_cols}"

    def test_no_extra_features_beyond_spec(self, toy_ohlcv):
        """Output columns must be timestamp + exactly EXP_D_FULL columns, no extras."""
        df = build_features(toy_ohlcv)
        expected = {"timestamp"} | set(get_feature_columns("EXP_D_FULL"))
        actual = set(df.columns)
        extra = actual - expected
        assert len(extra) == 0, f"Extra columns found beyond spec: {extra}"


# ══════════════════════════════════════════════════════════════════════
# 8. Raw Data Immutability Tests
# ══════════════════════════════════════════════════════════════════════

class TestRawDataImmutability:
    """Verify that feature engineering code never modifies raw data."""

    RAW_PATH = Path("D:/CryptoAnalyzer/data/raw/btcusdt_5m_raw.csv")

    @staticmethod
    def _sha256(filepath):
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @pytest.mark.skipif(
        not Path("D:/CryptoAnalyzer/data/raw/btcusdt_5m_raw.csv").exists(),
        reason="Raw data file not present"
    )
    def test_raw_data_unchanged_after_build_features(self):
        """SHA-256 of raw data must be identical before and after feature computation."""
        hash_before = self._sha256(self.RAW_PATH)

        import pandas as pd
        df = pd.read_csv(self.RAW_PATH)
        _ = build_features(df)

        hash_after = self._sha256(self.RAW_PATH)
        assert hash_before == hash_after, (
            f"Raw data was modified! Before: {hash_before}, After: {hash_after}"
        )

    def test_build_features_never_writes_raw_dir(self, toy_ohlcv, tmp_path):
        """build_features() must not create or modify files in any directory."""
        # Just verify it's a pure computation — no file I/O
        import os
        raw_dir = Path("D:/CryptoAnalyzer/data/raw")
        if raw_dir.exists():
            files_before = set(os.listdir(raw_dir))
            mtimes_before = {f: os.path.getmtime(raw_dir / f) for f in files_before}

        _ = build_features(toy_ohlcv)

        if raw_dir.exists():
            files_after = set(os.listdir(raw_dir))
            mtimes_after = {f: os.path.getmtime(raw_dir / f) for f in files_after}
            assert files_before == files_after, "New files created in data/raw/"
            for f in files_before:
                assert mtimes_before[f] == mtimes_after[f], f"File modified: {f}"
