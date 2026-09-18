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




---

## Decision 016: Phase 3 Feature Engineering Implementation & Warm-up Policy
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Implemented zero future look-ahead feature pipeline generating all 24 features for `EXP_D_FULL`. Defined explicit initialization policies:
  1. **EMA & ATR Warm-up:** All Exponential Moving Averages (`ema_12`, `ema_26`, MACD signal, `atr`, `rsi` components) explicitly use `adjust=False` and `min_periods=period` to ensure honest NaN propagation during warm-up and recursive seeding thereafter.
  2. **RSI Edge Cases:** RSI replaces theoretical 0/0 NaN (zero movement) with 50.0 (neutral), properly approaching 100 on absolute gains and 0 on absolute losses.
  3. **ATR Definition:** Uses `skipna=False` on True Range component max aggregation to correctly yield NaN when the shifted previous close is NaN.
  4. **Max Warm-up:** The total pipeline maximum warm-up is 96 rows, bounded by `rolling_vol_96` (which requires 1 prior row for returns + 96 rows for standard deviation).
- **Reason:** Ensures mathematically strict implementation of indicators without the data leakage or initialization bias common in unverified financial ML pipelines.


---

## Decision 017: Phase 4 Target Generation, Sequence Construction, & Scaling Policy
- **Date:** 2026-09-16
- **Status:** APPROVED
- **Decision:** Implemented the 1-hour-ahead target definitions and 3D sliding window sequence generator with the following methodology:
  1. **Target Formulas (Locked):**
     - Regression: `R(t, 12) = ((Close[t+12] - Close[t]) / Close[t]) * 100`
     - Classification: `D(t, 12) = 1 if R(t, 12) > 0 else 0`
     - Zero return maps to `D=0` (no positive movement).
  2. **Final H Rows:** The final 12 rows of any feature dataset cannot produce a valid target (no `Close[t+12]` exists). These rows are explicitly removed via `dropna()` on the NaN-shifted future close series â€” not by silent slicing. Resulting target-valid rows: 8544 - 12 = 8532.
  3. **Sequence Window:** W = 60. Each sequence X[i] covers indices [t-59 ... t] with shape (60, K). First valid sequence starts at index 59, last valid at index 8531, yielding 8473 sequences total before splitting.
  4. **Cross-Split Target Leakage Prevention:** Sequence assignment to a partition is determined by where the *future target observation* (index t+12) falls â€” not merely where the window end falls. Sequences whose `t+12` index falls in the validation region are assigned to validation; those in the test region go to test. This prevents a training-labeled window from consuming a future observation that belongs to the validation or test temporal partition.
  5. **Chronological Split (70/15/15):** Boundaries are computed in integer row indices on the 8544 fully populated feature rows:
     - `train_end_idx = int(8544 * 0.70) = 5980`
     - `val_end_idx = int(8544 * 0.85) = 7262`
     - No random shuffling. No k-fold cross-validation.
  6. **Scaler Policy:** `StandardScaler` is fit exclusively on the **2D feature rows** of the training partition (`features[0:train_end_idx]`). Validation and test feature rows are transformed using training statistics only. Targets are never included in scaler input. This avoids repeated weighting of overlapping 3D window timestamps that would occur with flattened sequence-based fitting.
  7. **Resulting Tensor Shapes (real dataset):**
     - `EXP_A_PRICE`: Train (5909, 60, 4) | Val (1282, 60, 4) | Test (1282, 60, 4)
     - `EXP_B_PRICE_VOL`: Train (5909, 60, 5) | Val (1282, 60, 5) | Test (1282, 60, 5)
     - `EXP_C_TECH_IND`: Train (5909, 60, 19) | Val (1282, 60, 19) | Test (1282, 60, 19)
     - `EXP_D_FULL`: Train (5909, 60, 24) | Val (1282, 60, 24) | Test (1282, 60, 24)
- **Reason:** Ensures a fully leakage-free, reproducible, chronologically valid sequence dataset ready for model training in subsequent phases.

---

### [2026-09-17] Decision 018: Phase 5 Baseline Model & Evaluation Design

- **Context:** Implementing non-deep-learning baselines (Naive, Ridge, Logistic, RandomForest, XGBoost) to establish benchmarks before Phase 6 DL models.
- **Decisions Made:**
  1. **3D Tensor Flattening Policy:** Baseline non-DL models require 2D matrix inputs `(N, features)`. Sliding window tensors `(N, 60, K)` are flattened row-wise into `(N, 60 * K)` feature vectors. This preserves all temporal sequence information within each window without discarding lookback steps.
  2. **Hyperparameter Locking Policy:** Fixed, deterministic hyperparameters with `random_state=42` across all models to establish a deterministic/reproducible configuration where practical:
     - Naive: Predicts zero-lag momentum return $R(t-1, 1)$ / direction sign
     - Ridge: `alpha=1.0`, `fit_intercept=True`
     - Logistic: `C=0.1`, `max_iter=1000`, `solver='lbfgs'`
     - RandomForest: `n_estimators=100`, `max_depth=None`, `min_samples_leaf=5`, `n_jobs=-1` (production execution)
     - XGBoost: `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`
  3. **MAPE Handling Strategy:** Relative percentage errors (MAPE) on target returns $R(t, 12)$ can spike artificially when actual returns approach 0%. MAPE computation handles division by zero safely using `np.where(abs(y_true) > 1e-8, abs((y_true - y_pred) / y_true), 0.0)` and reporting finite float values.
  4. **RandomForest Reproducibility in Unit Tests:** Parallel execution (`n_jobs=-1`) in scikit-learn RandomForest can cause non-deterministic floating-point reduction order variations (~$10^{-16}$). In production evaluation runner `n_jobs=-1` is used for multi-core speed, while unit tests evaluate reproducibility using `n_jobs=1` and `np.testing.assert_array_almost_equal(decimal=10)`.
  5. **Train-Only Model Fitting:** Baseline models are strictly fit on the `train` dataset partition (`X_train`, `y_train`). Validation (`X_val`) and test (`X_test`) partitions are evaluated strictly out-of-sample with zero parameter updating.
- **Reason:** Ensures transparent, reproducible, and leakage-free baseline benchmarks across all 4 experiment configurations (`EXP_A` through `EXP_D`).


---

### [2026-09-18] Decision 019: Phase 8 Regime Slope Threshold Calibration

- **Context:** RESEARCH.md Section 8 specifies theta_slope = 0.0005 as a *default starting point* for the normalized OLS slope threshold. The actual slope distribution for this BTC/USDT 5-minute dataset was found to have a maximum of ~0.000378 (far below 0.0005), meaning the documented constant would classify 100% of rows as Sideways — rendering regime segmentation uninformative.
- **Decision:**
  1. **Data-Adaptive Slope Threshold:** theta_slope is derived as the **50th percentile (median) of |slope|** computed over the **training partition only** (rows 0..5979). This mirrors the existing train-only policy for theta_vol_high (Decision 017).
  2. **Documentation of Documented Default:** The documented constant 0.0005 is preserved in RESEARCH.md as the original specification and in the classifier module as THETA_SLOPE. The derived value is reported separately as 	heta_slope_used in all output artifacts and the markdown report.
  3. **Train-Only Derivation:** Both theta_slope and theta_vol_high are computed exclusively from the training partition to prevent future-data leakage into regime labels used in out-of-sample evaluation.
  4. **Derived Thresholds (this dataset):** theta_slope_used ˜ 0.000040 (50th pctile of training abs(slope)); theta_vol_high ˜ 0.001896 (90th pctile of training rolling volatility).
- **Reason:** Ensures the regime segmentation produces meaningful, non-degenerate regime labels while maintaining strict leakage-free derivation from training data only. The adaptation is explicitly documented so results are fully reproducible.
