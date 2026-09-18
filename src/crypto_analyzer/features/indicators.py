"""
Feature engineering indicator computation module for CryptoAnalyzer.

Computes returns, technical indicators, and volatility metrics from OHLCV data.
Every function uses only information available at or before timestamp t (zero future look-ahead).
All rolling computations enforce min_periods to produce honest NaN during warm-up.
All EMA computations use adjust=False and min_periods=period explicitly.

Feature Reference Table
=======================

| Feature       | Function               | Input Columns  | Period | Warm-up (NaN rows) | Formula                                                      |
|---------------|------------------------|----------------|--------|---------------------|--------------------------------------------------------------|
| returns       | compute_returns        | close          | 1      | 1 (index 0)         | R_t = (Close_t - Close_{t-1}) / Close_{t-1}                 |
| log_returns   | compute_log_returns    | close          | 1      | 1 (index 0)         | r_t = ln(Close_t / Close_{t-1})                              |
| sma_12        | compute_sma            | close          | 12     | 11 (indices 0-10)   | SMA_n(t) = (1/n) * sum_{i=0}^{n-1} Close_{t-i}              |
| sma_24        | compute_sma            | close          | 24     | 23 (indices 0-22)   | SMA_n(t) = (1/n) * sum_{i=0}^{n-1} Close_{t-i}              |
| sma_96        | compute_sma            | close          | 96     | 95 (indices 0-94)   | SMA_n(t) = (1/n) * sum_{i=0}^{n-1} Close_{t-i}              |
| ema_12        | compute_ema            | close          | 12     | 11 (indices 0-10)   | EMA_n(t) = alpha*P_t + (1-alpha)*EMA_{t-1}, alpha=2/(n+1)   |
| ema_26        | compute_ema            | close          | 26     | 25 (indices 0-24)   | EMA_n(t) = alpha*P_t + (1-alpha)*EMA_{t-1}, alpha=2/(n+1)   |
| rsi_14        | compute_rsi            | close          | 14     | 14 (indices 0-13)   | RSI = 100 - 100/(1 + EMA_14(gains)/EMA_14(losses))          |
| macd          | compute_macd           | close          | 12,26  | 25 (indices 0-24)   | MACD = EMA_12 - EMA_26                                       |
| macd_signal   | compute_macd           | close          | 9      | 33 (indices 0-32)   | Signal = EMA_9(MACD) with min_periods=9                      |
| macd_hist     | compute_macd           | close          | -      | 33 (indices 0-32)   | Histogram = MACD - Signal                                    |
| bb_upper      | compute_bollinger_bands| close          | 20     | 19 (indices 0-18)   | SMA_20 + 2*sigma_20                                          |
| bb_lower      | compute_bollinger_bands| close          | 20     | 19 (indices 0-18)   | SMA_20 - 2*sigma_20                                          |
| bb_width      | compute_bollinger_bands| close          | 20     | 19 (indices 0-18)   | (Upper - Lower) / SMA_20                                     |
| bb_pct_b      | compute_bollinger_bands| close          | 20     | 19 (indices 0-18)   | (Close - Lower) / (Upper - Lower)                            |
| atr_14        | compute_atr            | high,low,close | 14     | 14 (indices 0-13)   | ATR = EMA_14(TR), TR=max(H-L, |H-Pc|, |L-Pc|)               |
| rolling_vol_12| compute_rolling_volatility | returns    | 12     | 12 (indices 0-11)   | std(R_{t-11}..R_t), but returns[0]=NaN → valid at idx 12     |
| rolling_vol_24| compute_rolling_volatility | returns    | 24     | 24 (indices 0-23)   | std(R_{t-23}..R_t), but returns[0]=NaN → valid at idx 24     |
| rolling_vol_96| compute_rolling_volatility | returns    | 96     | 96 (indices 0-95)   | std(R_{t-95}..R_t), but returns[0]=NaN → valid at idx 96     |
| atr_ratio     | compute_atr_ratio      | atr_14, close  | -      | 14 (from ATR)       | ATR_14 / Close                                               |
"""

import numpy as np
import pandas as pd


def compute_returns(close: pd.Series) -> pd.Series:
    """
    Compute 1-period simple returns.

    Formula:
        R_t = (Close_t - Close_{t-1}) / Close_{t-1}

    Input columns: close
    Period: 1
    Warm-up: 1 NaN row (index 0, no prior candle)

    Parameters
    ----------
    close : pd.Series
        Close price series.

    Returns
    -------
    pd.Series
        Simple return series. First value is NaN.
    """
    return close.pct_change(periods=1)


def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Compute 1-period log returns.

    Formula:
        r_t = ln(Close_t / Close_{t-1})

    Input columns: close
    Period: 1
    Warm-up: 1 NaN row (index 0)

    Parameters
    ----------
    close : pd.Series
        Close price series.

    Returns
    -------
    pd.Series
        Log return series. First value is NaN.
    """
    return np.log(close / close.shift(1))


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Simple Moving Average.

    Formula:
        SMA_n(t) = (1/n) * sum_{i=0}^{n-1} P_{t-i}

    Uses rolling(period, min_periods=period) — only data at or before t.
    First (period-1) values are NaN.

    Input columns: close (or any price series)
    Warm-up: (period - 1) NaN rows

    Parameters
    ----------
    series : pd.Series
        Input price series.
    period : int
        Window size for SMA.

    Returns
    -------
    pd.Series
        SMA series with NaN for first (period-1) rows.
    """
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average.

    Formula:
        EMA_n(t) = alpha * P_t + (1 - alpha) * EMA_n(t-1)
        where alpha = 2 / (n + 1)

    Implementation:
        pandas ewm(span=period, min_periods=period, adjust=False).mean()

    Initialization policy:
        - adjust=False: recursive EMA from the first non-NaN observation.
        - min_periods=period: first (period-1) values are NaN. The EMA is
          seeded at the (period)-th non-NaN observation, which becomes the
          initial EMA value, and subsequent values apply the recursive formula.

    Warm-up: (period - 1) NaN rows

    Parameters
    ----------
    series : pd.Series
        Input price series.
    period : int
        Span for EMA smoothing.

    Returns
    -------
    pd.Series
        EMA series with NaN for first (period-1) rows.
    """
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index using EMA-smoothed gains and losses.

    Formula:
        delta_t = Close_t - Close_{t-1}
        U_t = max(delta_t, 0)        (gain)
        D_t = max(-delta_t, 0)       (loss)
        avg_gain = EMA_n(U_t)  with adjust=False, min_periods=period
        avg_loss = EMA_n(D_t)  with adjust=False, min_periods=period
        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Input columns: close
    Period: 14 (default)
    Warm-up: `period` NaN rows (indices 0 to period-1).
        - close.diff() produces NaN at index 0.
        - EMA with min_periods=period requires `period` non-NaN gain/loss
          values. Since gains[0]=NaN, the first `period` non-NaN values
          span indices 1..period, producing the first valid EMA at index
          `period`. Combined with the diff NaN, RSI has `period` NaN rows.

    Edge-case behavior:
        - All gains (avg_loss=0): RS → +inf → RSI = 100.
        - All losses (avg_gain=0): RS = 0 → RSI = 0.
        - No movement (avg_gain=0 AND avg_loss=0): RS = 0/0 = NaN.
          We explicitly replace this with RSI = 50 (neutral) since
          zero price movement implies no directional bias.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    period : int
        RSI lookback period (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100]. First `period` values are NaN.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # EMA smoothing with explicit min_periods=period and adjust=False
    avg_gain = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    # Handle edge cases before division
    # Case 1: avg_loss == 0 and avg_gain > 0 → RS = inf → RSI = 100
    # Case 2: avg_gain == 0 and avg_loss > 0 → RS = 0 → RSI = 0
    # Case 3: avg_gain == 0 and avg_loss == 0 → RS = NaN → RSI = 50 (neutral)
    both_zero = (avg_gain == 0) & (avg_loss == 0)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Replace NaN from 0/0 with 50 (neutral RSI)
    rsi = rsi.where(~both_zero, 50.0)

    # Preserve NaN for warm-up rows (indices 0 to period-1)
    # The ewm min_periods already handles this, but we explicitly enforce
    # to guard against any edge-case where ewm might produce a value.
    rsi.iloc[:period] = np.nan

    return rsi


def compute_macd(close: pd.Series) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Formula:
        MACD(t) = EMA_12(t) - EMA_26(t)
        Signal(t) = EMA_9(MACD(t))  with min_periods=9, adjust=False
        Histogram(t) = MACD(t) - Signal(t)

    Input columns: close
    Periods: EMA_12 (fast), EMA_26 (slow), EMA_9 (signal)

    Warm-up behavior (follows from EMA initialization policy):
        - EMA_12: min_periods=12 → first 11 NaN
        - EMA_26: min_periods=26 → first 25 NaN
        - MACD line = EMA_12 - EMA_26: first 25 NaN (driven by EMA_26)
        - Signal = EMA_9(MACD) with min_periods=9: needs 9 non-NaN MACD
          values. First non-NaN MACD is at index 25, so first valid Signal
          is at index 25 + 9 - 1 = 33. → first 33 NaN.
        - Histogram = MACD - Signal: first 33 NaN (driven by Signal).

    No explicit warm-up row count is claimed beyond what follows from
    the EMA min_periods initialization policy described above.

    Parameters
    ----------
    close : pd.Series
        Close price series.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['macd', 'macd_signal', 'macd_hist'].
    """
    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    }, index=close.index)


def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Formula:
        SMA_20 = rolling(20, min_periods=20).mean(close)
        sigma_20 = rolling(20, min_periods=20).std(close, ddof=0)
        Upper(t) = SMA_20(t) + 2 * sigma_20(t)
        Lower(t) = SMA_20(t) - 2 * sigma_20(t)
        Width(t) = (Upper(t) - Lower(t)) / SMA_20(t)
        %B(t) = (Close_t - Lower(t)) / (Upper(t) - Lower(t))

    Input columns: close
    Period: 20 (default), std_dev multiplier: 2.0 (default)
    Warm-up: (period - 1) = 19 NaN rows

    Parameters
    ----------
    close : pd.Series
        Close price series.
    period : int
        Window for SMA and std dev (default 20).
    num_std : float
        Number of standard deviations (default 2.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['bb_upper', 'bb_lower', 'bb_width', 'bb_pct_b'].
        First (period-1) rows are NaN.
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std(ddof=0)

    upper = sma + num_std * rolling_std
    lower = sma - num_std * rolling_std
    width = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct_b": pct_b,
    }, index=close.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Average True Range.

    Formula:
        prev_close = Close_{t-1}  (shift(1), NaN at index 0)
        TR_t = max(High_t - Low_t, |High_t - prev_close|, |Low_t - prev_close|)
        ATR_n(t) = EMA_n(TR_t)  with adjust=False, min_periods=period

    Input columns: high, low, close
    Period: 14 (default)
    Warm-up: `period` NaN rows (indices 0 to period-1).
        - True Range has NaN at index 0 (from prev_close shift).
        - EMA with min_periods=period requires `period` non-NaN TR values.
          Since TR[0]=NaN, the first `period` non-NaN TR values span
          indices 1..period, producing the first valid ATR at index `period`.

    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    period : int
        ATR smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR values. First `period` values are NaN.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)

    # EMA smoothing with explicit min_periods and adjust=False
    atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()

    return atr


def compute_rolling_volatility(returns: pd.Series, period: int) -> pd.Series:
    """
    Compute rolling standard deviation of returns (return volatility).

    Formula:
        sigma_m(t) = std(R_{t-m+1}, ..., R_t) over m periods (ddof=1).

    Input: pre-computed simple return series (from compute_returns)
    Warm-up: Because returns[0]=NaN and rolling requires `period` non-NaN
        values, the first valid value appears at index `period`:
        - returns[0]=NaN, returns[1..period] = period non-NaN values
        - rolling(period, min_periods=period) first valid window is
          returns[1..period] at index `period`.
        So NaN for indices 0 to (period-1), first valid at index `period`.

    Parameters
    ----------
    returns : pd.Series
        Simple return series (already computed via compute_returns).
    period : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling volatility series.
    """
    return returns.rolling(window=period, min_periods=period).std(ddof=1)


def compute_atr_ratio(atr: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute normalized ATR ratio.

    Formula:
        ATR_Ratio(t) = ATR_14(t) / Close_t

    Input: pre-computed ATR series, close price series
    Warm-up: Inherits from ATR (first `period` rows NaN).

    Parameters
    ----------
    atr : pd.Series
        ATR values (from compute_atr).
    close : pd.Series
        Close price series.

    Returns
    -------
    pd.Series
        Normalized ATR ratio. NaN where ATR is NaN.
    """
    return atr / close
