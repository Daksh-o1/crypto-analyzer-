# CryptoAnalyzer Task Tracker (TASKS.md)

## COMPLETED (Phase 2)
- [x] Implement Binance BTC/USDT 5-minute historical candle downloader module (`BinanceDownloader`)
- [x] Implement dataset integrity & boundary validator (`DataValidator`):
  - [x] Strictly increasing timestamps
  - [x] Expected 5-minute intervals (300,000 ms)
  - [x] UTC normalization
  - [x] Duplicate timestamp detection
  - [x] Missing candle detection
  - [x] OHLC logic consistency check ($\text{High} \ge \max(\text{Open}, \text{Close})$, $\text{Low} \le \min(\text{Open}, \text{Close})$)
  - [x] Positive OHLC values check ($\text{Open}, \text{High}, \text{Low}, \text{Close} > 0$)
  - [x] Non-negative volume check ($\text{Volume} \ge 0$)
  - [x] NaN / Inf value detection
  - [x] Schema validation
  - [x] Date coverage verification
  - [x] Row count validation
  - [x] Dataset provenance logging
  - [x] File hash / checksum verification (SHA-256)
- [x] Store raw immutable datasets in `data/raw/btcusdt_5m_raw.csv` (8,640 rows, SHA-256: `a722751b...`)
- [x] Output dataset metadata summary report (`dataset_metadata.json`, `validation_report.json`)
- [x] Create CLI script `scripts/download_data.py`
- [x] Implement unit test suite `tests/test_data_pipeline.py` (13 passed)
- [x] Record Decision 015 in `DECISIONS.md`
- [x] Mark Phase 2 COMPLETE and STOP for Human Approval for Phase 3

## COMPLETED (Phase 3 — Feature Engineering)
- [x] Implement return & log-return generators ($R_t, r_t$, lag returns)
- [x] Implement technical indicators (SMA_12/24/96, EMA_12/26, RSI_14, MACD/Signal/Hist, Bollinger Upper/Lower/Width/%B, ATR_14)
- [x] Implement volatility feature calculators (rolling std dev of 5m returns over 12/24/96 periods, normalized ATR ratio)
- [x] Validate temporal alignment and NaN handling with zero future look-ahead bias

- [x] Mark Phase 3 COMPLETE and STOP for Human Approval for Phase 4

## COMPLETED (Phase 4 — Target & Sequence Pipeline)
- [x] Implement `compute_targets()` generating `R_t_12` and `D_t_12` via the locked formulas
- [x] Explicitly drop final H=12 rows with no valid future target
- [x] Implement `fit_transform_scaler()` fitting `StandardScaler` on training rows only
- [x] Implement `extract_3d_sequences()` with chronological sliding-window of shape (N, 60, K)
- [x] Prevent cross-split target leakage by assigning sequences via `t+12` realization index
- [x] Support all four experiment configurations: K = 4, 5, 19, 24
- [x] Implement `build_experiment_tensors()` orchestrator pipeline
- [x] Create `build_sequences.py` CLI and verify shapes on real 8544-row dataset
- [x] Wrote `tests/test_targets.py` (7 tests) and `tests/test_sequences.py` (4 tests)
- [x] Full test suite: 116 tests passing
- [x] Mark Phase 4 COMPLETE and STOP for Human Approval for Phase 5

## COMPLETED (Phase 5 — Baseline Models)
- [x] Implement Naive / Lag baseline model class (`NaiveBaseline`)
- [x] Implement Ridge Regressor baseline model class (`RidgeRegressorModel`)
- [x] Implement Logistic Classifier baseline model class (`LogisticModel`)
- [x] Implement Random Forest Regressor & Classifier model classes (`RandomForestModel`)
- [x] Implement XGBoost Regressor & Classifier model classes (`XGBoostModel`)
- [x] Implement experiment orchestration runner (`ExperimentRunner`) with $(N, 60, K) \to (N, 60 \times K)$ tensor flattening
- [x] Implement evaluation metrics computation (MAE, RMSE, MAPE, Accuracy, Precision, Recall, F1, Confusion Matrix)
- [x] Create CLI runner script `scripts/run_baselines.py`
- [x] Implement unit test suite `tests/test_baselines.py` (31/31 passed)
- [x] Execute complete real-data baseline experiment run generating JSON artifacts in `experiments/results/`
- [x] Verify raw data SHA-256 integrity remains untouched
- [x] Record Decision 018 in `DECISIONS.md`
- [x] Mark Phase 5 COMPLETE and STOP for Human Approval for Phase 6

## BACKLOG
- [ ] Phase 10: Build interactive Streamlit analyzer app
- [ ] Phase 11: Execute complete repository audit & clean environment test
- [ ] Phase 12: Write research paper deliverables

## BLOCKED
- None at present.

## COMPLETED
- [x] Phase 0 project foundation & environment setup (2026-09-16)
- [x] Phase 1 research foundation & scientific specification (2026-09-16)
- [x] Phase 2 BTC/USDT raw data pipeline & 14-point validation (2026-09-16)
- [x] Phase 3 feature engineering — 24 features, 84 tests, zero look-ahead (2026-09-16)
- [x] Phase 4 target & sequence pipeline — 3D tensors, 11 tests, 116 full suite (2026-09-16)
- [x] Phase 5 baseline models (2026-09-17)
- [x] Phase 6 deep learning models (2026-09-18)
- [x] Phase 7 feature ablation study (2026-09-18)
- [x] Phase 8 market regime analysis (2026-09-18)
- [x] Phase 9 statistical analysis & comparative reports (2026-09-19)
