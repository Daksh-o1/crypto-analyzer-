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
- **Decision:** Prediction targets are formulated as (1) Binary Directional Classification ($D(t, H) \in \{0, 1\}$) and (2) Percentage Return Regression ($R(t, H) \in \mathbb{R}$). Direct scalar future price ($P_{t+H}$) prediction is explicitly rejected.
- **Reason:** Predicting nominal price levels ($P_{t+H}$) introduces non-stationarity issues where price scale shifts over time (e.g. BTC at \$20,000 vs \$100,000). Percentage return and direction are scale-stable target representations suitable for time-series modeling.

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

---

## Decision 007: Controlled 4-Tier Feature Ablation Matrix (Phase 1)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Establish four distinct feature subsets for systematic ablation experiments:
  - `EXP_A_PRICE`: Raw OHLC prices (4 features)
  - `EXP_B_PRICE_VOL`: OHLC + Volume (5 features)
  - `EXP_C_TECH_IND`: Price + Volume + Technical Indicators (19 features)
  - `EXP_D_FULL`: Full Feature Set including Volatility metrics (24 features)
- **Reason:** Testing Hypothesis H1 requires isolated, incremental feature additions to prove whether technical indicators and volatility metrics add statistically significant predictive value.

---

## Decision 008: Multi-Model Benchmark Taxonomy (Phase 1)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Benchmark seven distinct model architectures spanning four model families:
  1. Naive Lag Baseline
  2. Ridge Linear / Logistic Regression
  3. Random Forest
  4. XGBoost
  5. Stacked LSTM (2-layer)
  6. Stacked GRU (2-layer)
  7. 1D-CNN (Convolutional sequence model)
- **Reason:** Evaluates Hypothesis H2 and SQ3 by comparing linear, non-linear tree, and recurrent/convolutional deep learning models under identical temporal input constraints ($W=60$ lookback window).

---

## Decision 009: Mathematical Market Regime Classification (Phase 1)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Define market regimes using a rolling 24-hour window ($W_{\text{regime}}=288$ candles) with two quantitative metrics: (1) OLS normalized price slope $S_t$, and (2) return standard deviation $\sigma_t$. Segment performance into Bullish, Bearish, Sideways, and High Volatility regimes.
- **Reason:** Evaluates Hypothesis H3 and SQ4 by testing whether model performance degrades or improves under specific market dynamics rather than relying solely on aggregate test metrics.

---

## Decision 010: Fit-Only-On-Train Normalization Protocol (Phase 1)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Require all scalers (`StandardScaler` / `MinMaxScaler`) to be fitted strictly on the 70% training partition, with parameters applied unchanged to validation (15%) and test (15%) partitions.
- **Reason:** Prevents look-ahead bias and future distribution leakage, addressing a major flaw in public ML repositories.

---

## Decision 011: Paired Statistical Significance Testing Protocol (Phase 1)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Implement non-parametric paired Wilcoxon signed-rank tests across prediction error and hit series to validate model and feature performance gains at the $\alpha = 0.05$ significance level.
- **Reason:** Ensures reported improvements in directional accuracy or MAE are statistically meaningful rather than random noise.

---

## Decision 012: Modernization & Overhaul of Reference Repository (Phase 1)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Re-architect the benchmark from the reference repository (`khuangaf/CryptocurrencyPrediction`) by upgrading from Python 3.6 / Keras 2.x to Python 3.11 / PyTorch / Scikit-learn, fixing look-ahead scaler fits, replacing nominal price target with scale-stable returns/directions suitable for time-series modeling, enforcing chronological splits, and introducing unit tests.
- **Reason:** Elevates the baseline code from a flawed demonstration script into a publishable, reproducible scientific research framework.

---

## Decision 013: Prediction Horizon Resolution (H=12) & Scale-Stable Target Terminology Revision (Phase 1 Revision)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:**
  1. **Primary Prediction Horizon ($H$):** Formally set primary prediction horizon to $H = 12$ candles (60 minutes = 1 hour ahead) computed over 5-minute BTC/USDT OHLCV input candles with historical context lookback window $W = 60$ candles (5 hours). Return regression target is defined as $R(t, H) = \left(\frac{P_{t+H} - P_t}{P_t}\right) \times 100$ and directional classification target as $D(t, H) = 1$ if $R(t, H) > 0$ else $0$.
  2. **Target Terminology Revision:** Removed claims that cryptocurrency return series are strictly stationary. Standardized target terminology across all specifications to "scale-stable return and directional targets suitable for time-series modeling".
- **Reason:** Explicitly aligns input lookback context (5 hours = 60 candles) with a meaningful intraday forecast horizon (1 hour = 12 candles) and corrects theoretical claims regarding return series stationarity in financial econometrics.
- **Alternatives Considered:** Predicting 1-step 5-minute horizon ($H=1$) (too noisy, high spread-to-return ratio for trade analysis); claiming strict statistical stationarity (theoretically inaccurate).

---

## Decision 014: Master Phase Plan Alignment & Standardized Data Validation Protocol Revision
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:**
  1. **Phase Plan Synchronization:** Updated `phases.md` to explicitly align Phase 1, Phase 4, and Phase 9 specifications with `RESEARCH.md` and `DECISIONS.md` (Decision 011 and Decision 013). Fixed Phase 1 and Phase 4 target definitions to 5-minute candles, $W = 60$ (5-hour context window), $H = 12$ (1-hour forecast horizon), return regression $R(t, 12) = \left(\frac{\text{Close}[t+12] - \text{Close}[t]}{\text{Close}[t]}\right) \times 100$, and directional classification $D(t, 12) = 1$ if $R(t, 12) > 0$ else $0$.
  2. **Comprehensive Data Validation Protocol (Phase 2):** Expanded Phase 2 data validator specifications to require checks for strictly increasing timestamps, expected 5-minute intervals (300,000 ms), UTC normalization, duplicate detection, missing candles detection, OHLC consistency ($\text{High} \ge \max(\text{Open}, \text{Close})$, $\text{Low} \le \min(\text{Open}, \text{Close})$), positive OHLC values, non-negative volume ($\text{Volume} \ge 0$), NaN/Inf value detection, schema validation, date coverage, row count, dataset provenance, and file checksums (SHA-256).
  3. **Statistical Testing Guidance (Phase 9):** Clarified that statistical testing in Phase 9 must follow Decision 011, applying paired Wilcoxon signed-rank tests where appropriate to paired model evaluation designs rather than blindly across all metrics.
  4. **Validation Command Standard:** Standardized all validation commands across `phases.md` to explicitly invoke the virtual environment Python interpreter: `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest ...`.
- **Reason:** Eliminates discrepancies between master phase plan documentation, research specification docs, and execution environment requirements prior to commencing Phase 2.

---

## Decision 015: Binance BTC/USDT Raw Data Acquisition & Validation Pipeline (Phase 2)
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Implemented `BinanceDownloader` and `DataValidator` modules in `src/crypto_analyzer/data/`. Acquired 8,640 raw 5-minute OHLCV candles (30 days of market depth: 2026-08-17T09:45:00Z to 2026-09-16T09:40:00Z) from Binance Spot API (`https://api.binance.com/api/v3/klines`), stored immutably as `D:\CryptoAnalyzer\data\raw\btcusdt_5m_raw.csv` with SHA-256 hash `a722751b2929acdcae48cb875be4c3d4904d17e99d45ea0b3f6d9a67a4cf1eb8`. Verified zero data leakage, zero feature engineering, zero target creation, zero sequence tensors, zero model training, and zero missing data fabrication.
- **Reason:** Provides the immutable, reproducible empirical dataset required for subsequent feature calculation and model sequence generation while fulfilling all 14 data quality and provenance criteria specified in `phases.md`.


