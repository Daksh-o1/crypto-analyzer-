"""
Feature engineering pipeline orchestrator for CryptoAnalyzer.

Assembles all indicator computations into a unified feature DataFrame
and provides experiment-aware column selection for the ablation matrix.

Warm-Up Analysis
================
The maximum warm-up across all 24 features is driven by rolling_vol_96:
  - compute_returns produces NaN at index 0.
  - rolling(96, min_periods=96) on returns needs 96 non-NaN values.
  - Since returns[0]=NaN, the window returns[1..96] gives 96 non-NaN at index 96.
  - Therefore rolling_vol_96 has 96 NaN rows (indices 0-95), first valid at index 96.

This exceeds SMA_96 (95 NaN rows, first valid at index 95) because rolling_vol_96
operates on returns (which themselves have a 1-row warm-up) rather than raw prices.

MAX_WARM_UP_ROWS = 96 (conservative maximum across all features).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

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


# ──────────────────────────────────────────────────────────────────────
# Experiment → Feature Column Definitions
# Exactly matches docs/experiments.md Section 1.1
# ──────────────────────────────────────────────────────────────────────

EXPERIMENT_FEATURES: Dict[str, List[str]] = {
    "EXP_A_PRICE": [
        "open",
        "high",
        "low",
        "close",
    ],
    "EXP_B_PRICE_VOL": [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ],
    "EXP_C_TECH_IND": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_12",
        "sma_24",
        "sma_96",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_pct_b",
        "atr_14",
    ],
    "EXP_D_FULL": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_12",
        "sma_24",
        "sma_96",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_pct_b",
        "atr_14",
        "rolling_vol_12",
        "rolling_vol_24",
        "rolling_vol_96",
        "atr_ratio",
        "log_returns",
    ],
}

# Expected feature counts per experiment (used for validation)
EXPECTED_FEATURE_COUNTS: Dict[str, int] = {
    "EXP_A_PRICE": 4,
    "EXP_B_PRICE_VOL": 5,
    "EXP_C_TECH_IND": 19,
    "EXP_D_FULL": 24,
}

# Maximum warm-up rows across all indicators.
# Driven by rolling_vol_96: returns[0]=NaN + rolling(96) → 96 NaN rows.
MAX_WARM_UP_ROWS = 96


def get_feature_columns(experiment_id: str) -> List[str]:
    """
    Return the exact ordered list of feature column names for a given experiment.

    Parameters
    ----------
    experiment_id : str
        One of 'EXP_A_PRICE', 'EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL'.

    Returns
    -------
    List[str]
        Ordered list of feature column names.

    Raises
    ------
    ValueError
        If experiment_id is not recognized.
    """
    if experiment_id not in EXPERIMENT_FEATURES:
        raise ValueError(
            f"Unknown experiment_id '{experiment_id}'. "
            f"Valid IDs: {list(EXPERIMENT_FEATURES.keys())}"
        )
    columns = EXPERIMENT_FEATURES[experiment_id]
    expected_count = EXPECTED_FEATURE_COUNTS[experiment_id]
    assert len(columns) == expected_count, (
        f"Internal error: {experiment_id} has {len(columns)} columns, "
        f"expected {expected_count}"
    )
    return list(columns)


def get_warm_up_rows() -> int:
    """
    Return the number of initial rows that contain NaN due to indicator warm-up.

    The maximum warm-up across all features is 96 rows, driven by
    rolling_vol_96 (returns warm-up + 96-period rolling window).

    Individual features have different warm-up periods — tests verify
    actual NaN behavior for each feature independently.

    Returns
    -------
    int
        Conservative warm-up row count (96).
    """
    return MAX_WARM_UP_ROWS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 24 features from raw OHLCV data.

    Reads a raw OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
    and computes all features needed for EXP_D_FULL. No future data is accessed.
    No targets are created. No scaling is applied.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with columns: timestamp, open, high, low, close, volume.
        Must be sorted chronologically with strictly increasing timestamps.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with columns:
        [timestamp, open, high, low, close, volume,
         sma_12, sma_24, sma_96, ema_12, ema_26,
         rsi_14, macd, macd_signal, macd_hist,
         bb_upper, bb_lower, bb_width, bb_pct_b, atr_14,
         rolling_vol_12, rolling_vol_24, rolling_vol_96,
         atr_ratio, log_returns]

    Raises
    ------
    ValueError
        If required input columns are missing or data is not sorted.
    """
    # ── Input validation ──────────────────────────────────────────────
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < 2:
        raise ValueError(f"DataFrame must have at least 2 rows, got {len(df)}")

    # Verify chronological ordering
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("DataFrame timestamps must be strictly increasing (chronological order)")

    # ── Work on a copy to preserve immutability of input ──────────────
    result = df.copy()

    # ── 1. Simple Returns (needed for rolling volatility) ─────────────
    simple_returns = compute_returns(result["close"])

    # ── 2. Log Returns ────────────────────────────────────────────────
    result["log_returns"] = compute_log_returns(result["close"])

    # ── 3. Simple Moving Averages ─────────────────────────────────────
    result["sma_12"] = compute_sma(result["close"], period=12)
    result["sma_24"] = compute_sma(result["close"], period=24)
    result["sma_96"] = compute_sma(result["close"], period=96)

    # ── 4. Exponential Moving Averages (min_periods=period, adjust=False) ──
    result["ema_12"] = compute_ema(result["close"], period=12)
    result["ema_26"] = compute_ema(result["close"], period=26)

    # ── 5. RSI (EMA-smoothed, min_periods=period, adjust=False) ───────
    result["rsi_14"] = compute_rsi(result["close"], period=14)

    # ── 6. MACD (EMA warm-up follows from min_periods policy) ─────────
    macd_df = compute_macd(result["close"])
    result["macd"] = macd_df["macd"]
    result["macd_signal"] = macd_df["macd_signal"]
    result["macd_hist"] = macd_df["macd_hist"]

    # ── 7. Bollinger Bands ────────────────────────────────────────────
    bb_df = compute_bollinger_bands(result["close"], period=20, num_std=2.0)
    result["bb_upper"] = bb_df["bb_upper"]
    result["bb_lower"] = bb_df["bb_lower"]
    result["bb_width"] = bb_df["bb_width"]
    result["bb_pct_b"] = bb_df["bb_pct_b"]

    # ── 8. Average True Range (EMA with min_periods=period, adjust=False) ──
    result["atr_14"] = compute_atr(result["high"], result["low"], result["close"], period=14)

    # ── 9. Rolling Volatility (std dev of simple returns) ─────────────
    result["rolling_vol_12"] = compute_rolling_volatility(simple_returns, period=12)
    result["rolling_vol_24"] = compute_rolling_volatility(simple_returns, period=24)
    result["rolling_vol_96"] = compute_rolling_volatility(simple_returns, period=96)

    # ── 10. Normalized ATR Ratio ──────────────────────────────────────
    result["atr_ratio"] = compute_atr_ratio(result["atr_14"], result["close"])

    # ── Validate output column counts ─────────────────────────────────
    for exp_id, expected_count in EXPECTED_FEATURE_COUNTS.items():
        feature_cols = EXPERIMENT_FEATURES[exp_id]
        present = [c for c in feature_cols if c in result.columns]
        if len(present) != expected_count:
            raise RuntimeError(
                f"Feature count mismatch for {exp_id}: "
                f"expected {expected_count}, found {len(present)} "
                f"(missing: {set(feature_cols) - set(present)})"
            )

    return result


def get_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of computed features including NaN counts and warm-up info.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame from build_features().

    Returns
    -------
    Dict
        Summary dictionary with feature statistics.
    """
    feature_cols = get_feature_columns("EXP_D_FULL")
    summary = {
        "total_rows": len(df),
        "warm_up_rows": get_warm_up_rows(),
        "usable_rows": len(df) - get_warm_up_rows(),
        "feature_counts": {
            exp_id: len(cols)
            for exp_id, cols in EXPERIMENT_FEATURES.items()
        },
        "nan_counts": {},
        "inf_counts": {},
    }

    for col in feature_cols:
        if col in df.columns:
            summary["nan_counts"][col] = int(df[col].isna().sum())
            summary["inf_counts"][col] = int(np.isinf(df[col].dropna()).sum())

    return summary
