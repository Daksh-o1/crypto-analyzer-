"""
Feature engineering package for CryptoAnalyzer.
Provides technical indicator computation and experiment-aware feature pipeline.
"""

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
    EXPERIMENT_FEATURES,
    EXPECTED_FEATURE_COUNTS,
    MAX_WARM_UP_ROWS,
    build_features,
    get_feature_columns,
    get_feature_summary,
    get_warm_up_rows,
)

__all__ = [
    # Indicators
    "compute_returns",
    "compute_log_returns",
    "compute_sma",
    "compute_ema",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_rolling_volatility",
    "compute_atr_ratio",
    # Pipeline
    "build_features",
    "get_feature_columns",
    "get_feature_summary",
    "get_warm_up_rows",
    "EXPERIMENT_FEATURES",
    "EXPECTED_FEATURE_COUNTS",
    "MAX_WARM_UP_ROWS",
]
