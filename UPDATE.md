# CryptoAnalyzer Chronological Development Log (UPDATE.md)

---

## [2026-09-16] — Phase 0 Foundation Setup

**Date:** 2026-09-16  
**Phase:** Phase 0 (Project Foundation)  
**Work Completed:**
1. Verified D: drive availability and disk space (72+ GB available).
2. Created project root directory strictly at `D:\CryptoAnalyzer`.
3. Initialized Git repository.
4. Created Python 3.11.15 virtual environment on D: at `D:\CryptoAnalyzer\.venv` using `uv`.
5. Created complete modular directory structure (`src/`, `docs/`, `data/`, `experiments/`, `models/`, `reports/`, `tests/`, `scripts/`, `notebooks/`).
6. Configured `.gitignore` and created `.gitkeep` files in empty tracked subdirectories.
7. Configured `requirements.txt` and `pyproject.toml`.
8. Installed baseline testing & data science stack (`pytest`, `pytest-cov`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `requests`, `pyyaml`).
9. Formulated core documentation (`README.md`, `phases.md`, `UPDATE.md`, `SUMMARY.md`, `RESEARCH.md`, `DECISIONS.md`, `TASKS.md`, `docs/*`).
10. Built test infrastructure (`tests/conftest.py`, `tests/test_environment.py`).
11. Executed validation test suite with 100% pass rate.

**Files Changed:**
- `D:\CryptoAnalyzer\.gitignore`
- `D:\CryptoAnalyzer\requirements.txt`
- `D:\CryptoAnalyzer\pyproject.toml`
- `D:\CryptoAnalyzer\README.md`
- `D:\CryptoAnalyzer\phases.md`
- `D:\CryptoAnalyzer\RESEARCH.md`
- `D:\CryptoAnalyzer\DECISIONS.md`
- `D:\CryptoAnalyzer\TASKS.md`
- `D:\CryptoAnalyzer\SUMMARY.md`
- `D:\CryptoAnalyzer\UPDATE.md`
- `D:\CryptoAnalyzer\src\crypto_analyzer\__init__.py`
- `D:\CryptoAnalyzer\tests\__init__.py`
- `D:\CryptoAnalyzer\tests\conftest.py`
- `D:\CryptoAnalyzer\tests\test_environment.py`
- `D:\CryptoAnalyzer\docs\architecture.md`
- `D:\CryptoAnalyzer\docs\dataset.md`
- `D:\CryptoAnalyzer\docs\methodology.md`
- `D:\CryptoAnalyzer\docs\experiments.md`
- `D:\CryptoAnalyzer\docs\reproducibility.md`

**Tests Performed:**
- Ran `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/`
- Verified Python version (3.11.15 >= 3.10)
- Verified package importability (`crypto_analyzer.__version__ == "0.1.0"`)
- Verified project drive location (`D:`)
- Verified presence of all mandatory directories and documentation files.

**Results:**
- All 5 test cases in `tests/test_environment.py` passed cleanly.

**Problems Encountered & Resolved:**
- Default `python` path on C: drive had an uninstalled launcher reference. Resolved by creating the Python virtual environment via `uv` pointing directly to Python 3.11.15 on `D:\CryptoAnalyzer\.venv`.

**Remaining Issues:**
- None for Phase 0.

**Next Action:**
- Phase 0 Complete. Started Phase 1 upon human approval.

---

## [2026-09-16] — Phase 1 Research Foundation & Methodology Revision

**Date:** 2026-09-16  
**Phase:** Phase 1 (Research Foundation)  
**Work Completed:**
1. Finalized research problem statement and scientific motivation addressing flaws in existing financial ML approaches (look-ahead scaler fitting, random CV data leakage, non-stationary nominal price targets).
2. Formulated primary research question and 4 sub-questions (SQ1 baseline benchmarking, SQ2 feature ablation, SQ3 deep learning vs tree ensembles, SQ4 market regime shifts).
3. Developed 4 formal scientific hypotheses (H1 Feature Efficacy, H2 Architecture Efficacy, H3 Market Regime Conditioning, H4 Target Metric Divergence).
4. Defined formal mathematical targets for 1-hour percentage return regression ($R(t, H=12)$) and binary directional classification ($D(t, H=12)$) over a 5-hour context window ($W=60$).
5. **Research Methodology Revisions (Decision 013):**
   - Removed claims that return series are strictly "stationary", revising terminology across all documentation to "scale-stable return and directional targets suitable for time-series modeling".
   - Explicitly resolved prediction horizon: input context $W = 60$ candles (5 hours) $\rightarrow$ primary prediction horizon $H = 12$ candles (60 minutes / 1 hour ahead), defined as $R(t, H) = \frac{\text{Close}(t+H) - \text{Close}(t)}{\text{Close}(t)} \times 100$ and $D(t, H) = 1$ if $R(t, H) > 0$ else $0$.
6. Specified dataset methodology (Binance BTC/USDT 5-minute spot candles, storage on D: drive `data/raw/`, 70% train / 15% val / 15% test chronological partitioning, zero-lookahead fit-only-on-train normalization).
7. Established 4-tier feature ablation matrix (`EXP_A_PRICE`, `EXP_B_PRICE_VOL`, `EXP_C_TECH_IND`, `EXP_D_FULL`) across 24 indicators and features.
8. Defined model taxonomy comparing 7 models across 4 families (Naive Lag, Ridge Linear/Logistic Regression, Random Forest, XGBoost, Stacked LSTM, Stacked GRU, 1D-CNN) with 3D sliding sequence formulation ($W=60$ lookback window).
9. Formulated mathematical market regime classification methodology using rolling 24-hour window ($W_{\text{regime}}=288$) for Bullish, Bearish, Sideways, and High Volatility regimes.
10. Specified classification metrics ($DA$, Precision, Recall, Macro F1, Confusion Matrix) and regression metrics (MAE, RMSE, MAPE) with paired Wilcoxon signed-rank significance testing protocol.
11. Documented threats to validity (look-ahead bias, target leakage, data leakage, overfitting, regime shift) and project limitations (single asset pair BTC/USDT, technical feature boundary, frictionless market assumption).
12. Detailed comprehensive critique and modernization mapping relative to reference repository (`khuangaf/CryptocurrencyPrediction`).
13. Recorded finalized Decisions 007 through 014 in `DECISIONS.md`.
14. **Master Phase Plan Alignment (Decision 014):**
    - Updated `phases.md` to align Phase 1, Phase 4, and Phase 9 target descriptions and statistical testing guidelines with `RESEARCH.md` and `DECISIONS.md`.
    - Expanded Phase 2 data validator tasks and acceptance criteria (timestamps, 5-min intervals, UTC, duplicates, missing candles, OHLC logic, positive OHLC, non-negative volume, NaN/Inf, schema, date coverage, row count, provenance, SHA-256 checksum).
    - Standardized all phase validation commands to explicitly invoke `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest ...`.
15. Created and updated unit test suite `tests/test_research_spec.py` to programmatically validate research specifications and horizon/terminology constraints.
16. Executed validation test suite and verified 100% pass rate (8 passed).

**Files Changed:**
- `D:\CryptoAnalyzer\RESEARCH.md`
- `D:\CryptoAnalyzer\DECISIONS.md`
- `D:\CryptoAnalyzer\docs\methodology.md`
- `D:\CryptoAnalyzer\docs\experiments.md`
- `D:\CryptoAnalyzer\docs\reproducibility.md`
- `D:\CryptoAnalyzer\tests\test_research_spec.py`
- `D:\CryptoAnalyzer\phases.md`
- `D:\CryptoAnalyzer\UPDATE.md`
- `D:\CryptoAnalyzer\SUMMARY.md`
- `D:\CryptoAnalyzer\TASKS.md`
- `D:\CryptoAnalyzer\README.md`

**Tests Performed:**
- Executed `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/`
- Verified 8 tests passed in 0.12s (5 environment tests + 3 research specification tests).

**Results:**
- All acceptance criteria for Phase 1 satisfied 100%.

**Problems Encountered & Resolved:**
- Corrected target terminology from "stationary" to "scale-stable return targets suitable for time-series modeling" and explicitly fixed prediction horizon to $H=12$ (1 hour) over $W=60$ (5 hours) lookback in Decision 013. Synchronized `phases.md` and expanded Phase 2 data validation criteria in Decision 014.

**Remaining Issues:**
- None for Phase 1.

---

## Phase 1 Completion Report

**Status:** COMPLETE  

**Implemented:**
- Finalized scientific research foundation in `RESEARCH.md` and `docs/methodology.md`.
- Formulated primary RQ, SQ1-SQ4, and Hypotheses H1-H4.
- Defined formal scale-stable targets ($R(t, H=12)$ 1-hour return regression, $D(t, H=12)$ directional classification over $W=60$ 5-hour lookback).
- Established 4-tier feature ablation matrix (`EXP_A` through `EXP_D`).
- Defined 7-model taxonomy matrix and sequence tensor shape ($N, W=60, K$).
- Defined 4 mathematical market regimes (Bullish, Bearish, Sideways, High Volatility).
- Formulated evaluation metrics and Wilcoxon signed-rank significance testing protocol.
- Documented threats to validity, project limitations, and detailed reference repository comparison.
- Added Decisions 007 to 014 to `DECISIONS.md`.
- Synchronized `phases.md` with explicit $W=60, H=12$ targets, expanded Phase 2 data validation protocol, and standardized virtualenv validation commands.
- Explicitly enumerated exact feature column lists in `docs/experiments.md` for `EXP_A_PRICE` (4), `EXP_B_PRICE_VOL` (5), `EXP_C_TECH_IND` (19), and `EXP_D_FULL` (24).
- Created and passed test suite `tests/test_research_spec.py`.

**Tests:**
- Executed `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/` -> 8 passed in 0.15s.

**Validation:**
- Verified all acceptance criteria for Phase 1.
- Confirmed zero dataset downloads, zero model training, zero dashboard building, and zero Phase 2 start.

**Next Phase:** Phase 2 (Data Pipeline) — STARTED upon human approval.

---

## [2026-09-16] — Phase 2 Data Pipeline Execution

**Date:** 2026-09-16  
**Phase:** Phase 2 (Data Pipeline)  
**Work Completed:**
1. Built reproducible Binance historical 5-minute candle downloader module `BinanceDownloader` in `src/crypto_analyzer/data/downloader.py`.
2. Implemented `DataValidator` in `src/crypto_analyzer/data/validator.py` enforcing all 14 mandatory quality and integrity checks (schema validation, NaN/Inf detection, strictly increasing timestamps, duplicate timestamp detection, 5-minute step intervals, missing candle gap reporting, UTC normalization, OHLC logic consistency, positive prices, non-negative volume, date coverage, row count, provenance, and SHA-256 file checksum).
3. Built metadata logging data structures `DatasetMetadata` and `ValidationReport` in `src/crypto_analyzer/data/metadata.py`.
4. Executed live data acquisition from Binance Spot API (`https://api.binance.com/api/v3/klines`) via `scripts/download_data.py`.
5. Acquired 8,640 raw 5-minute OHLCV candles (30 days of market depth: 2026-08-17T09:45:00Z to 2026-09-16T09:40:00Z) and saved raw dataset immutably as `D:\CryptoAnalyzer\data\raw\btcusdt_5m_raw.csv`.
6. Verified dataset integrity: 0 missing candles, 0 duplicate timestamps, 0 OHLC violations, 0 non-positive prices, 0 negative volumes, 0 NaN/Inf values.
7. Calculated and saved SHA-256 file checksum (`a722751b2929acdcae48cb875be4c3d4904d17e99d45ea0b3f6d9a67a4cf1eb8`), `dataset_metadata.json`, and `validation_report.json` under `D:\CryptoAnalyzer\data\raw\`.
8. Created comprehensive pytest suite `tests/test_data_pipeline.py` (13 test cases) covering downloader initialization, mock generation, CSV persistence, all 14 validator checks, error detection, JSON serialization, and zero-leakage code audit.
9. Recorded Decision 015 in `DECISIONS.md`.
10. Confirmed strict compliance with research constraints: zero technical indicators calculated, zero model features created, zero targets constructed, zero sequence tensors generated, zero models trained, zero UI built, zero Phase 3 work performed.

**Files Created / Changed:**
- `D:\CryptoAnalyzer\src\crypto_analyzer\data\__init__.py`
- `D:\CryptoAnalyzer\src\crypto_analyzer\data\downloader.py`
- `D:\CryptoAnalyzer\src\crypto_analyzer\data\validator.py`
- `D:\CryptoAnalyzer\src\crypto_analyzer\data\metadata.py`
- `D:\CryptoAnalyzer\scripts\download_data.py`
- `D:\CryptoAnalyzer\tests\test_data_pipeline.py`
- `D:\CryptoAnalyzer\data\raw\btcusdt_5m_raw.csv`
- `D:\CryptoAnalyzer\data\raw\dataset_metadata.json`
- `D:\CryptoAnalyzer\data\raw\validation_report.json`
- `D:\CryptoAnalyzer\DECISIONS.md`
- `D:\CryptoAnalyzer\phases.md`
- `D:\CryptoAnalyzer\UPDATE.md`
- `D:\CryptoAnalyzer\SUMMARY.md`
- `D:\CryptoAnalyzer\TASKS.md`
- `D:\CryptoAnalyzer\README.md`

**Tests Performed:**
- Executed `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/` -> 21 passed in 3.27s (5 environment + 3 research spec + 13 data pipeline tests).

**Results:**
- All 14 data quality and provenance validation checks passed 100%.

**Problems Encountered & Resolved:**
- Resolved SyntaxError in `metadata.py` classmethod definitions.
- Configured pythonpath / sys.path in `scripts/download_data.py` for direct CLI execution.

**Remaining Issues:**
- None for Phase 2.

---

## Phase 2 Completion Report

**Status:** COMPLETE  

**Implemented:**
- Reproducible Binance BTC/USDT 5-minute candle downloader `BinanceDownloader`.
- Immutable raw data storage under `D:\CryptoAnalyzer\data\raw\btcusdt_5m_raw.csv` (8,640 rows, SHA-256: `a722751b...`).
- Comprehensive `DataValidator` enforcing 14 data quality checks (timestamps, 5m steps, UTC, duplicates, gaps, OHLC logic, positive prices, volume, NaN/Inf, schema, date coverage, row count, provenance, SHA-256).
- Metadata and validation report logging (`dataset_metadata.json`, `validation_report.json`).
- CLI script `scripts/download_data.py`.
- Automated test suite `tests/test_data_pipeline.py` (13 passed).
- Added Decision 015 to `DECISIONS.md`.

**Tests:**
- Executed `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/` -> 21 passed in 3.27s.

**Validation:**
- Verified all acceptance criteria for Phase 2.
- Verified zero technical indicators calculated, zero model features created, zero targets created, zero sequence tensors generated, zero model training, zero UI built, zero Phase 3 start.

**Next Phase:** Phase 3 (Feature Engineering) — Awaiting Approval.






---

## [2026-09-16] — Phase 3 Feature Engineering

**Date:** 2026-09-16  
**Phase:** Phase 3 (Feature Engineering)  
**Work Completed:**
1. Created indicators.py containing zero look-ahead functions for returns, log returns, SMA, EMA, RSI, MACD, Bollinger Bands, ATR, rolling volatility, and ATR ratio.
2. Enforced explicit min_periods=period and djust=False on all EMA-based calculations (EMA, MACD, RSI components, ATR).
3. Created pipeline.py implementing uild_features to compute 24 features (for EXP_D_FULL) in a single pass.
4. Defined column mappings for EXP_A_PRICE, EXP_B_PRICE_VOL, EXP_C_TECH_IND, and EXP_D_FULL.
5. Created CLI script compute_features.py which loads tcusdt_5m_raw.csv, runs pipeline, outputs stats/counts, and saves tcusdt_5m_features.csv.
6. Built comprehensive test suite (	est_features.py) with 84 tests covering mathematical correctness, zero look-ahead via future-row mutation, per-feature NaN propagation, and expected column counts.
7. Verified pipeline against real dataset (8640 rows -> 8544 usable rows post-warm-up).
8. Verified zero sequence tensors, target construction, or model operations occurred.
9. Added Decision 016 mapping out explicit warm-up and NaN edge-case policies.

**Files Created / Changed:**
- D:\CryptoAnalyzer\src\crypto_analyzer\features\__init__.py
- D:\CryptoAnalyzer\src\crypto_analyzer\features\indicators.py
- D:\CryptoAnalyzer\src\crypto_analyzer\features\pipeline.py
- D:\CryptoAnalyzer\scripts\compute_features.py
- D:\CryptoAnalyzer\tests\test_features.py
- D:\CryptoAnalyzer\data\processed\btcusdt_5m_features.csv
- D:\CryptoAnalyzer\data\processed\feature_summary.json
- D:\CryptoAnalyzer\DECISIONS.md
- D:\CryptoAnalyzer\phases.md
- D:\CryptoAnalyzer\UPDATE.md
- D:\CryptoAnalyzer\SUMMARY.md
- D:\CryptoAnalyzer\TASKS.md

**Tests Performed:**
- Executed D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/ -> 105 passed.

**Results:**
- All features correctly calculated with exact expected counts. Raw data immutability confirmed.

- Phase 3 Complete. Proceeded to Phase 4 upon human approval.

---

## [2026-09-16] — Phase 4 Target & Sequence Pipeline

**Date:** 2026-09-16
**Phase:** Phase 4 (Target & Sequence Pipeline)
**Work Completed:**
1. Created `targets.py` implementing `compute_targets()` with the locked formulas:
   - Regression: `R(t, 12) = ((Close[t+12] - Close[t]) / Close[t]) * 100`
   - Classification: `D(t, 12) = 1 if R(t, 12) > 0 else 0`
   - Final 12 rows with no valid `Close[t+12]` are explicitly removed via `dropna()`.
2. Created `sequences.py` implementing:
   - `fit_transform_scaler()`: fits `StandardScaler` on 2D training feature rows only; transforms val/test with train parameters.
   - `extract_3d_sequences()`: builds chronological (60, K) windows; assigns sequences by target realization index (`t+12`) to prevent cross-split target leakage.
3. Created `pipeline.py` implementing `build_experiment_tensors()` orchestrating warm-up removal, target generation, scaling, and 3D tensor building for all 4 experiment layouts.
4. Created `scripts/build_sequences.py` CLI; validated on real 8544-row dataset.
5. Created `tests/test_targets.py` (7 tests) and `tests/test_sequences.py` (4 tests).
6. Recorded Decision 017 in `DECISIONS.md`.

**Row Progression:**
- 8640 raw rows → 8544 fully-populated (drop 96 warm-up NaNs) → 8532 target-valid (drop final 12) → 8473 total sequences

**Split Boundaries (70/15/15 chronological):**
- `train_end_idx = 5980`, `val_end_idx = 7262`

**Tensor Shapes (verified on real dataset):**
- `EXP_A_PRICE`:    Train (5909, 60, 4)  | Val (1282, 60, 4)  | Test (1282, 60, 4)
- `EXP_B_PRICE_VOL`:Train (5909, 60, 5)  | Val (1282, 60, 5)  | Test (1282, 60, 5)
- `EXP_C_TECH_IND`: Train (5909, 60, 19) | Val (1282, 60, 19) | Test (1282, 60, 19)
- `EXP_D_FULL`:     Train (5909, 60, 24) | Val (1282, 60, 24) | Test (1282, 60, 24)

**Files Created / Changed:**
- `D:\CryptoAnalyzer\src\crypto_analyzer\preprocessing\__init__.py`
- `D:\CryptoAnalyzer\src\crypto_analyzer\preprocessing\targets.py`
- `D:\CryptoAnalyzer\src\crypto_analyzer\preprocessing\sequences.py`
- `D:\CryptoAnalyzer\src\crypto_analyzer\preprocessing\pipeline.py`
- `D:\CryptoAnalyzer\scripts\build_sequences.py`
- `D:\CryptoAnalyzer\tests\test_targets.py`
- `D:\CryptoAnalyzer\tests\test_sequences.py`
- `D:\CryptoAnalyzer\data\processed\tensors\*.npz` (4 experiment tensor archives)
- `D:\CryptoAnalyzer\data\processed\tensors\sequence_summary.json`
- `D:\CryptoAnalyzer\DECISIONS.md` (Decision 016 fixed, Decision 017 added)
- `D:\CryptoAnalyzer\phases.md`
- `D:\CryptoAnalyzer\TASKS.md`
- `D:\CryptoAnalyzer\UPDATE.md`
- `D:\CryptoAnalyzer\SUMMARY.md`

**Tests Performed:**
- `pytest tests/test_targets.py tests/test_sequences.py -v` → 11 passed
- `pytest tests/ -v` → 116 passed (full suite)

**Phase 4 Acceptance Criteria — ALL PASS:**
- [x] Targets exactly match `R(t,12)` and `D(t,12)`
- [x] W=60 and H=12 correctly implemented
- [x] All four experiment feature configurations supported (K=4,5,19,24)
- [x] Sequence dimensions correct
- [x] Sequence/target timestamps correctly aligned
- [x] Future observations cannot enter X
- [x] Final horizon rows explicitly excluded
- [x] Chronological 70/15/15 split passes tests
- [x] No train/val/test index overlap (cross-split leakage prevented via `t+12` boundary)
- [x] Scaler is train-only fitted; targets excluded from scaling
- [x] Raw data unchanged (no writes to `data/raw/`)
- [x] Phase 4 tests pass
- [x] Full suite passes
- [x] Documentation updated
- [x] No Phase 5+ implementation

**Next Action:**
- Phase 4 Complete.

---

## [2026-09-17] — Phase 5 Baseline Models

**Date:** 2026-09-17  
**Phase:** Phase 5 (Baseline Models)  
**Work Completed:**
1. Implemented baseline models in `src/crypto_analyzer/models/baselines.py`:
   - `NaiveBaseline` (predicts last return $R(t-1, 1)$ for regression; directional sign for classification)
   - `RidgeRegressorModel` (L2 regularized linear regression, alpha=1.0)
   - `LogisticModel` (L2 regularized logistic regression, C=0.1, solver='lbfgs')
   - `RandomForestModel` (Random Forest Regressor & Classifier, n_estimators=100, min_samples_leaf=5, n_jobs=-1)
   - `XGBoostModel` (XGBoost Regressor & Classifier, n_estimators=200, max_depth=4, learning_rate=0.05, tree_method='hist')
2. Implemented `ExperimentRunner` orchestration engine in `src/crypto_analyzer/models/experiment_runner.py`:
   - Flattens $(N, 60, K)$ input 3D sequence tensors into 2D $(N, 60 \times K)$ representations for baseline ML models.
   - Evaluates all 5 models across all 4 experiment feature configs (`EXP_A_PRICE`, `EXP_B_PRICE_VOL`, `EXP_C_TECH_IND`, `EXP_D_FULL`).
   - Computes regression metrics: MAE, RMSE, MAPE.
   - Computes classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
   - Saves individual experiment JSON artifacts and aggregated `all_baselines_results.json`.
3. Created CLI evaluation script `scripts/run_baselines.py`.
4. Implemented unit test suite in `tests/test_baselines.py` (31 unit tests covering model interface, tensor flattening, reproducibility, metric computation, output schema, missing input handling, zero-variance handling).
5. Executed complete real-data benchmark evaluation across all 20 experiment-model combinations.

**Files Created / Changed:**
- `D:\CryptoAnalyzer\src\crypto_analyzer\models\baselines.py` [NEW]
- `D:\CryptoAnalyzer\src\crypto_analyzer\models\experiment_runner.py` [NEW]
- `D:\CryptoAnalyzer\src\crypto_analyzer\models\__init__.py` [NEW]
- `D:\CryptoAnalyzer\scripts\run_baselines.py` [NEW]
- `D:\CryptoAnalyzer\tests\test_baselines.py` [NEW]
- `D:\CryptoAnalyzer\requirements.txt` (added `xgboost>=2.0.0`)
- `D:\CryptoAnalyzer\experiments\results\EXP_A_PRICE_baselines.json` [NEW]
- `D:\CryptoAnalyzer\experiments\results\EXP_B_PRICE_VOL_baselines.json` [NEW]
- `D:\CryptoAnalyzer\experiments\results\EXP_C_TECH_IND_baselines.json` [NEW]
- `D:\CryptoAnalyzer\experiments\results\EXP_D_FULL_baselines.json` [NEW]
- `D:\CryptoAnalyzer\experiments\results\all_baselines_results.json` [NEW]
- `D:\CryptoAnalyzer\DECISIONS.md` (Added Decision 018)
- `D:\CryptoAnalyzer\phases.md` (Marked Phase 5 COMPLETE)
- `D:\CryptoAnalyzer\TASKS.md` (Updated task items)
- `D:\CryptoAnalyzer\SUMMARY.md` (Updated current phase status)
- `D:\CryptoAnalyzer\UPDATE.md` (Logged Phase 5 entry)

**Test Results:**
- `pytest tests/test_baselines.py` → **31 passed** in 13.07s
- `pytest tests/` → **147 passed** (full project suite, 100% pass)

**Phase 5 Baseline Performance Summary (Test Partition - 1,282 samples):**

| Experiment | Model | Reg MAE | Reg RMSE | Cls Acc | Cls F1 |
|---|---|---|---|---|---|
| **EXP_A_PRICE** (K=4) | Naive | **0.2053** | **0.3150** | 0.5250 | 0.5186 |
| | Ridge | 0.2078 | 0.3155 | 0.5406 | 0.5342 |
| | Logistic | 0.2078 | 0.3155 | 0.5406 | 0.5342 |
| | RandomForest | 0.2104 | 0.3188 | 0.5398 | 0.5332 |
| | XGBoost | 0.2141 | 0.3235 | **0.5640** | **0.5491** |
| **EXP_B_PRICE_VOL** (K=5) | Naive | **0.2053** | **0.3150** | 0.5250 | 0.5186 |
| | Ridge | 0.2079 | 0.3156 | 0.5242 | 0.5159 |
| | Logistic | 0.2079 | 0.3156 | 0.5242 | 0.5159 |
| | RandomForest | 0.2114 | 0.3195 | **0.5257** | **0.5158** |
| | XGBoost | 0.2155 | 0.3253 | 0.5195 | 0.5057 |
| **EXP_C_TECH_IND** (K=19) | Naive | **0.2053** | **0.3150** | 0.5250 | 0.5186 |
| | Ridge | 0.2127 | 0.3204 | 0.5312 | 0.5304 |
| | Logistic | 0.2127 | 0.3204 | **0.5312** | **0.5304** |
| | RandomForest | 0.2126 | 0.3203 | 0.5273 | 0.5126 |
| | XGBoost | 0.2227 | 0.3341 | 0.5062 | 0.5011 |
| **EXP_D_FULL** (K=24) | Naive | **0.2053** | **0.3150** | 0.5250 | 0.5186 |
| | Ridge | 0.2133 | 0.3211 | 0.5359 | 0.5334 |
| | Logistic | 0.2133 | 0.3211 | **0.5359** | **0.5334** |
| | RandomForest | 0.2129 | 0.3203 | 0.5101 | 0.4907 |
| | XGBoost | 0.2241 | 0.3343 | 0.5031 | 0.4938 |

**Phase 5 Acceptance Criteria — ALL PASS:**
- [x] All 5 baseline models implemented (Naive, Ridge, Logistic, RandomForest, XGBoost)
- [x] All 4 experiment feature configurations benchmarked (EXP_A, EXP_B, EXP_C, EXP_D)
- [x] Inputs properly flattened from $(N, 60, K)$ to $(N, 60 \times K)$
- [x] Regression target $R(t, 12)$ evaluated with MAE, RMSE, MAPE
- [x] Classification target $D(t, 12)$ evaluated with Accuracy, Precision, Recall, F1, Confusion Matrix
- [x] Strict chronological train/validation/test split respected; zero leakage
- [x] Models fit on training partition only
- [x] Structured JSON metric outputs produced and verified in `experiments/results/`
- [x] 31/31 baseline unit tests pass; 147/147 full project unit tests pass
- [x] Raw data SHA-256 hash verified unchanged
- [x] Documentation updated (`UPDATE.md`, `DECISIONS.md`, `phases.md`, `TASKS.md`, `SUMMARY.md`)
- [x] Phase 6 deep learning implementation was NOT started

**Next Action:**
- Phase 5 Complete. STOP. Wait for explicit human approval before Phase 6.

---

## [2026-09-18] — Phase 6 Deep Learning Models Execution Complete

**Date:** 2026-09-18  
**Phase:** Phase 6 (Deep Learning Models)  
**Work Completed:**
1. Implemented modular deep learning architectures (`LSTMModel`, `GRUModel`, `CNN1DModel`) and a unified `DLTrainer` with Early Stopping, Checkpointing, and adaptive LR scheduling.
2. Verified multi-threading constraints on Windows CPU causing deadlocks, setting `torch.set_num_threads(1)` for stable single-thread deterministic execution.
3. Added robust epoch-level logging to `DLTrainer` to accurately monitor lengthy training durations (up to 2.2 hours per model without silent stalls).
4. Conducted full robust matrix execution comprising 24 combinations (3 architectures × 4 experiments × 2 tasks).
5. All 24 configurations successfully completed full training. Best epochs were automatically determined via Early Stopping (`patience=7`, `max_epochs=50`).
6. Verified no smoke-test results were accidentally marked as final results (smoke checkpoints explicitly deleted and rerun in full).
7. Maintained `Phase 5` baseline integrity, Phase 4 tensors, and raw data SHA256 integrity check.
8. Persisted state to `experiments/results/dl_matrix_status.json` ensuring reproducibility.

**Verification Checklist:**
- [x] All 24 models successfully trained on the Phase 4 60-step sequence tensors.
- [x] Evaluation rigorously utilized `torch.no_grad()` restricting test-set evaluation strictly to final inference.
- [x] Result artifacts securely saved to `experiments/results/`. Checkpoints saved to `models/checkpoints/`.
- [x] Phase 7 was NOT started.

**Next Action:**
- Phase 6 Complete. STOP. Wait for explicit human approval before Phase 7 (Feature Ablation Study).
---

## [2026-09-18] — Phase 7 Feature Ablation Study

**Date:** 2026-09-18  
**Phase:** Phase 7 (Feature Ablation Study)  
**Work Completed:**
1. Developed `scripts/run_ablation_analysis.py` to aggregate results from Phase 5 and Phase 6 JSON artifacts into a unified `ablation_summary.json`.
2. Generated `ablation_regression_comparison.md` and `ablation_classification_comparison.md` in `reports/tables/` detailing absolute metric differences (\u0394) between baseline (EXP_A_PRICE) and augmented feature sets (EXP_B, EXP_C, EXP_D) for 8 models.
3. Implemented Phase 7 unit tests in `tests/test_ablation.py` verifying metric integrity, missing models handling, schema validation, raw dataset immutability, and explicit distinction between `f1_binary` and `f1_macro`.
4. Verified that no models were retrained during this phase and that all existing 60-window 3D sequences and original evaluations remained unchanged.
5. All models (naive, ridge, logistic, random_forest, xgboost, LSTM, GRU, 1D-CNN) and experiments (A, B, C, D) correctly mapped and evaluated.

**Next Action:**
- Phase 7 Complete. STOP. Wait for explicit human approval before Phase 8 (Market Regime Analysis).

---

## [2026-09-18] - Phase 8 Market Regime Analysis Complete

**Date:** 2026-09-18
**Phase:** Phase 8 (Market Regime Analysis)
**Work Completed:**
1. Implemented modular market-regime classifier in `src/crypto_analyzer/regimes/classifier.py`:
   - Rolling OLS-slope and return-volatility computed with W_regime = 288 candles (24 hours).
   - Regime classification precedence: High Volatility first, then Bullish, Bearish, Sideways.
   - Both theta_vol_high and theta_slope derived strictly from training partition (train-only, rows 0..5979).
   - Decision 019 added to DECISIONS.md documenting data-adaptive theta_slope calibration.
2. Implemented regime analysis pipeline in `src/crypto_analyzer/regimes/analysis.py`:
   - Loads processed feature CSV, drops warm-up rows, aligns Close series, computes full-dataset regimes.
   - Reports regime statistics exclusively on the test partition (out-of-sample).
3. Wrote CLI script `scripts/run_regime_analysis.py`.
4. Generated artifacts:
   - experiments/results/regime_analysis_summary.json
   - experiments/results/test_regime_labels.csv
   - reports/tables/regime_analysis_report.md
5. Test partition regime distribution (test N=1282):
   - Sideways: 752 (58.7%)
   - Bearish: 299 (23.3%)
   - Bullish: 231 (18.0%)
   - High Volatility: 0 (0.0%)
   - NOTE: Zero High Volatility is a valid observed result indicating the test period was characterised by relatively low rolling volatility compared to the training period.
6. 36 Phase-8-specific tests written and passing.
7. Raw data SHA-256 verified unchanged.
8. Phase 4-7 artifacts verified unchanged.
9. Baseline predictions deterministically reconstructed (Phase 5 objects not persisted).
10. DL predictions loaded from existing Phase 6 checkpoints (no retraining).
11. Regime-wise model performance output generated successfully. High Volatility regime appropriately registers null/unavailable metrics due to 0 samples.

**Next Action:**
- Phase 8 COMPLETE (RESEARCH-READY). STOP. Wait for explicit human approval before Phase 9 (Experiment Analysis).
