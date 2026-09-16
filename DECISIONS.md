# CryptoAnalyzer Decision Log (DECISIONS.md)

This log records major architectural, methodological, and experimental decisions made during the project lifecycle.

---

## Decision 001: Project Storage Location (D: Drive Only)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** The entire project, including virtual environment (`.venv`), code, datasets (`data/`), model checkpoints (`models/`), caches, and reports, must reside exclusively on `D:\CryptoAnalyzer`.
- **Reason:** Enforces Rule 1 (D Drive Only) to preserve system drive storage, guarantee clean path isolation, and prevent fragmentation across C: and D: drives.
- **Alternatives Considered:** Storing virtual environment on C: AppData (Rejected per project mandate).
- **Trade-offs:** Requires explicit path invocation (`D:\CryptoAnalyzer\.venv\Scripts\python.exe`) when calling Python commands from shell contexts.

---

## Decision 002: Primary Asset Pair (BTC/USDT)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** BTC/USDT spot market data from Binance was selected as the primary research dataset.
- **Reason:** BTC/USDT provides the highest liquidity, longest historical depth, highest market capitalization, and minimal venue risk, serving as the benchmark asset for cryptocurrency market microstructure studies.
- **Alternatives Considered:** ETH/USDT, Multi-asset baskets (Deferred to future work in Phase 12).
- **Trade-offs:** Conclusions specifically reflect BTC market dynamics and may require re-calibration for low-cap altcoins.

---

## Decision 003: 5-Minute Candle Granularity
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** 5-minute OHLCV candles were chosen as the primary temporal timeframe.
- **Reason:** 5-minute resolution captures intraday volatility, order flow dynamics, and micro-structure trends without introducing the extreme high-frequency noise and tick-data storage overhead of 1-second/tick data.
- **Alternatives Considered:** 1-minute (too noisy/large storage), 1-hour (too few samples for deep sequence models per year).

---

## Decision 004: Dual Prediction Targets (Directional Classification + Return Regression)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Prediction targets are formulated as (1) Binary Directional Classification ($D_{t+1} \in \{0, 1\}$) and (2) Percentage Return Regression ($R_{t+1} \in \mathbb{R}$). Direct scalar future price ($P_{t+1}$) prediction is explicitly rejected.
- **Reason:** Predicting nominal price levels ($P_{t+1}$) introduces non-stationarity issues where price scale shifts over time (e.g. BTC at \$20,000 vs \$100,000). Percentage return and direction are stationary target representations essential for valid temporal modeling.

---

## Decision 005: Chronological Data Splitting & Leakage Prevention
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** A strict 70% Train / 15% Validation / 15% Test chronological split is enforced. Scalers and feature transformers are fitted **exclusively** on the training partition.
- **Reason:** Random shuffling or standard k-fold cross-validation on time-series data creates severe data leakage across temporal boundaries, invalidating backtest metrics.

---

## Decision 006: Python 3.11 & Modern ML Dependency Stack
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Use Python 3.11 with modern maintained packages (`pandas`, `scikit-learn`, `PyTorch` in Phase 6, `pytest`).
- **Reason:** The baseline reference repository (khuangaf/CryptocurrencyPrediction) relies on obsolete Python 3.6/Keras 2.x versions. Modernizing the stack ensures security, speed, and long-term research reproducibility.
