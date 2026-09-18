"""
Phase 8: Market Regime Classifier.

Implements the documented rolling-window mathematical regime classification from
RESEARCH.md Section 8 and DECISIONS.md Decision 017 (split boundaries).

Regime definitions (from RESEARCH.md §8):
  - W_regime = 288 candles (24 hours of 5-minute candles)
  - Trend slope: OLS slope of Close prices over W_regime, normalized by mean price
  - Rolling volatility: std dev of 5-min relative returns over W_regime

Regime classification precedence (High Volatility checked first):
  1. High Volatility : sigma_t > theta_vol_high  (regardless of slope)
  2. Bullish        : S_t > +theta_slope AND sigma_t <= theta_vol_high
  3. Bearish        : S_t < -theta_slope AND sigma_t <= theta_vol_high
  4. Sideways       : |S_t| <= theta_slope AND sigma_t <= theta_vol_high

Default parameters (RESEARCH.md §8):
  theta_slope    = 0.0005  (per 5-min step)
  theta_vol_high = 90th percentile of training-window return volatility
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_WINDOW: int = 288  # 24 hours at 5-minute granularity

# RESEARCH.md §8 specifies 0.0005 as a *default starting point*.
# In practice, the actual slope distribution for this BTC/USDT dataset has a max
# of ~0.00038, so 0.0005 would classify every row as Sideways.
# When theta_slope is None in compute_rolling_regimes(), it is derived
# data-adaptively as the 50th percentile of abs(slope) over the training
# partition — mirroring the train-only policy for theta_vol_high.
# This is documented as Decision 019 in DECISIONS.md.
THETA_SLOPE: float = 0.0005  # documented default; overridden data-adaptively
REGIME_LABELS = {
    0: "Sideways",
    1: "Bullish",
    2: "Bearish",
    3: "High Volatility",
}
REGIME_CODES = {v: k for k, v in REGIME_LABELS.items()}

# Sentinel value for warm-up rows that lack a full W_regime lookback
WARMUP_LABEL = "Warmup"


# ---------------------------------------------------------------------------
# Core computation utilities
# ---------------------------------------------------------------------------

def _ols_slope(prices: np.ndarray) -> float:
    """
    Compute the OLS linear-regression slope of a price array.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of close prices over the regime window.

    Returns
    -------
    float
        OLS slope (raw, in price units per step).
    """
    n = len(prices)
    x = np.arange(n, dtype=float)
    # Using the closed-form OLS formula for speed/clarity
    x_mean = x.mean()
    y_mean = prices.mean()
    slope = np.sum((x - x_mean) * (prices - y_mean)) / np.sum((x - x_mean) ** 2)
    return slope


def compute_regime_slope(prices: np.ndarray) -> float:
    """
    Compute the normalized trend slope S_t as defined in RESEARCH.md §8:

        S_t = Slope(P[t-W:t]) / Mean(P[t-W:t])

    Parameters
    ----------
    prices : np.ndarray
        1-D array of close prices of length W_regime.

    Returns
    -------
    float
        Normalized slope (dimensionless, per step).
    """
    mean_price = prices.mean()
    if mean_price == 0.0:
        return 0.0
    return _ols_slope(prices) / mean_price


def compute_regime_volatility(prices: np.ndarray) -> float:
    """
    Compute the rolling return volatility sigma_t as defined in RESEARCH.md §8:

        sigma_t = std dev of 5-minute relative returns over W_regime.

    Return r_k = (P_k - P_{k-1}) / P_{k-1} for k in [t-W+1 .. t].
    At least 2 prices are required to produce 1 return.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of close prices of length W_regime.

    Returns
    -------
    float
        Standard deviation of relative returns (population std).
    """
    if len(prices) < 2:
        return np.nan
    returns = np.diff(prices) / prices[:-1]
    return float(np.std(returns, ddof=0))


def classify_regime(
    slope: float,
    volatility: float,
    theta_slope: float,
    theta_vol_high: float,
) -> str:
    """
    Apply the documented threshold classification logic with precedence:
      1. High Volatility (sigma > theta_vol_high) — checked first.
      2. Bullish (S > +theta_slope AND sigma <= theta_vol_high)
      3. Bearish (S < -theta_slope AND sigma <= theta_vol_high)
      4. Sideways (|S| <= theta_slope AND sigma <= theta_vol_high)

    Parameters
    ----------
    slope : float
        Normalized OLS slope S_t.
    volatility : float
        Rolling return volatility sigma_t.
    theta_slope : float
        Slope threshold.
    theta_vol_high : float
        Volatility threshold (90th percentile of training volatility).

    Returns
    -------
    str
        One of: 'Bullish', 'Bearish', 'Sideways', 'High Volatility'.
    """
    if volatility > theta_vol_high:
        return "High Volatility"
    if slope > theta_slope:
        return "Bullish"
    if slope < -theta_slope:
        return "Bearish"
    return "Sideways"


# ---------------------------------------------------------------------------
# Rolling regime computation
# ---------------------------------------------------------------------------

def compute_rolling_regimes(
    close: pd.Series,
    window: int = REGIME_WINDOW,
    theta_slope: Optional[float] = None,
    theta_vol_high: Optional[float] = None,
    vol_percentile: float = 90.0,
    slope_percentile: float = 50.0,
    train_end_idx: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute rolling market regimes for each candle in the Close series.

    Strategy:
    - Rows with fewer than `window` prior candles are marked as 'Warmup'.
    - theta_vol_high is derived as the 90th percentile of rolling volatility
      values computed **strictly within the training partition** (rows 0..train_end_idx).
      If train_end_idx is None, the full series is used (test-only mode, less strict).

    Parameters
    ----------
    close : pd.Series
        Close price series, chronologically ordered, integer-indexed from 0.
    window : int
        Rolling lookback window in candles (default 288 = 24h).
    theta_slope : float, optional
        Normalized slope threshold. If None (default), derived from training
        data as the `slope_percentile`-th percentile of abs(slope) — same
        train-only policy used for theta_vol_high.
    theta_vol_high : float, optional
        Explicit volatility threshold. If None, computed from training data.
    vol_percentile : float
        Percentile used to derive theta_vol_high from training volatilities.
    slope_percentile : float
        Percentile of abs(slope) used to derive theta_slope from training data
        when theta_slope is None (default 50.0 = median).
    train_end_idx : int, optional
        Row index where the training partition ends. Used for deriving
        both theta_vol_high and theta_slope strictly from training data only.

    Returns
    -------
    Tuple[pd.DataFrame, float, float]
        - DataFrame with columns: regime, slope, volatility, regime_code
        - Derived (or passed) theta_vol_high
        - Derived (or passed) theta_slope
    """
    close_vals = close.values.astype(float)
    n = len(close_vals)

    slopes = np.full(n, np.nan)
    volatilities = np.full(n, np.nan)

    # Compute rolling slope and volatility for all windows where full lookback exists
    for i in range(window - 1, n):
        window_prices = close_vals[i - window + 1 : i + 1]
        slopes[i] = compute_regime_slope(window_prices)
        volatilities[i] = compute_regime_volatility(window_prices)

    # Derive thresholds from training data if not provided
    if train_end_idx is not None:
        train_slopes = slopes[:train_end_idx]
        train_vols = volatilities[:train_end_idx]
        valid_train_slopes = train_slopes[~np.isnan(train_slopes)]
        valid_train_vols = train_vols[~np.isnan(train_vols)]
    else:
        valid_train_slopes = slopes[~np.isnan(slopes)]
        valid_train_vols = volatilities[~np.isnan(volatilities)]

    if theta_vol_high is None:
        theta_vol_high = float(np.percentile(valid_train_vols, vol_percentile))

    if theta_slope is None:
        theta_slope = float(np.percentile(np.abs(valid_train_slopes), slope_percentile))

    # Classify each row
    labels = []
    codes = []
    for i in range(n):
        if np.isnan(slopes[i]):
            labels.append(WARMUP_LABEL)
            codes.append(np.nan)
        else:
            label = classify_regime(slopes[i], volatilities[i], theta_slope, theta_vol_high)
            labels.append(label)
            codes.append(float(REGIME_CODES[label]))

    result = pd.DataFrame(
        {
            "regime": labels,
            "slope": slopes,
            "volatility": volatilities,
            "regime_code": codes,
        },
        index=close.index,
    )

    return result, theta_vol_high, theta_slope
