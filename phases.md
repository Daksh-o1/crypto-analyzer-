# CryptoAnalyzer Master Project Plan (phases.md)

This document tracks the phased development lifecycle for the **CryptoAnalyzer** research project.

Strict Workflow:
`PHASE N` → `Implement` → `Tests/Checks` → `Review` → `Fix` → `Verify Acceptance Criteria` → `Update Docs` → `Mark COMPLETE` → `STOP` → `WAIT FOR HUMAN APPROVAL` → `PHASE N+1`

---

## Phase Overview Table

| Phase | Title | Objective | Status |
|---|---|---|---|
| Phase 0 | Project Foundation | Establish environment, structure, git, docs, and test harness on D: | **COMPLETE** |
| Phase 1 | Research Foundation | Finalize formal scientific definitions, hypotheses, and experimental design | **COMPLETE** |
| Phase 2 | Data Pipeline | Build reproducible BTC/USDT data acquisition, validation, and storage | **COMPLETE** |
| Phase 3 | Feature Engineering | Compute return, volume, indicator, and volatility features without leakage | NOT_STARTED |
| Phase 4 | Target & Sequence Pipeline | Create 3D sequence tensors, targets, chronologically split, fit train scalers | NOT_STARTED |
| Phase 5 | Baseline Models | Implement Naive, Linear, Random Forest, and XGBoost baselines | NOT_STARTED |
| Phase 6 | Deep Learning Models | Implement LSTM, GRU, and 1D-CNN modular architectures & training loops | NOT_STARTED |
| Phase 7 | Feature Ablation Study | Execute comparative Experiments A, B, C, D across feature sets | NOT_STARTED |
| Phase 8 | Market Regime Analysis | Formally segment performance across Bull, Bear, Sideways, & Volatile regimes | NOT_STARTED |
| Phase 9 | Experiment Analysis | Comprehensive statistical comparison, tables, and figures | NOT_STARTED |
| Phase 10 | Crypto Analyzer Application | Build interactive Streamlit analyzer UI with strict disclaimer | NOT_STARTED |
| Phase 11 | Final Validation | Full test suite, clean-environment re-run, leakage audit, code cleanup | NOT_STARTED |
| Phase 12 | Research Deliverables | Final paper/report, methodology summary, and reproducible artifacts | NOT_STARTED |

---

## Phase 0: Project Foundation

- **Objective:** Create a professional project skeleton, virtual environment, git repository, test setup, and master documentation strictly on D: drive (`D:\CryptoAnalyzer`).
- **Tasks:**
  1. Inspect D: drive existence and storage.
  2. Create `D:\CryptoAnalyzer` root directory.
  3. Initialize Git repository.
  4. Create complete directory hierarchy (`src/`, `docs/`, `data/`, `experiments/`, `models/`, `reports/`, `tests/`, `scripts/`, `notebooks/`).
  5. Create virtual environment on D: (`D:\CryptoAnalyzer\.venv`) with Python 3.11.
  6. Configure `requirements.txt` and `pyproject.toml`.
  7. Install basic test dependencies (`pytest`, `pytest-cov`, `pandas`, `numpy`, `scikit-learn`, `pyyaml`).
  8. Create initial documentation files (`README.md`, `phases.md`, `UPDATE.md`, `SUMMARY.md`, `RESEARCH.md`, `DECISIONS.md`, `TASKS.md`, `docs/*`).
  9. Implement basic test infrastructure (`tests/conftest.py`, `tests/test_environment.py`).
  10. Run pytest suite and verify 100% pass rate.
- **Acceptance Criteria:**
  - Project exists strictly on D: drive (`D:\CryptoAnalyzer`).
  - Virtual environment located on D: (`D:\CryptoAnalyzer\.venv`).
  - Git repository initialized.
  - All directory structures and `.gitkeep` files present.
  - Pytest suite executes and passes without errors.
  - All 7 core documentation files created and populated.
- **Validation Command:**
  `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/`
- **Status:** **COMPLETE**

---

## Phase 1: Research Foundation

- **Objective:** Finalize the scientific definition of the project, including formal scale-stable targets, feature categories, model taxonomy, evaluation metrics, and threats to validity.
- **Tasks:**
  - Formulate precise research questions and hypotheses H1-H4.
  - Define formal mathematical definitions for 1-hour forecast horizon ($H = 12$ candles = 60 minutes) percentage return regression $R(t, 12)$ and binary directional classification $D(t, 12)$ using 5-minute candles over a 5-hour lookback window ($W = 60$ candles).
  - Detail experimental feature sets (Price-only, Price+Volume, Technical Indicators, Volatility).
  - Document market regime classification methodology ($W_{\text{regime}}=288$ candles).
  - Specify threats to validity and reproducibility criteria.
  - Document relationship to reference repository `khuangaf/CryptocurrencyPrediction`.
- **Deliverables:** Complete `RESEARCH.md`, `DECISIONS.md`, `docs/methodology.md`, `docs/experiments.md`, `docs/reproducibility.md`, and `tests/test_research_spec.py`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_research_spec.py`
- **Status:** **COMPLETE**

---

## Phase 2: Data Pipeline

- **Objective:** Build reproducible BTC/USDT data acquisition, validation, raw data persistence, and integrity metadata logging.
- **Tasks:**
  - Implement Binance historical 5-minute candle data downloader module.
  - Store immutable raw datasets in `data/raw/`.
  - Build comprehensive data validator enforcing strict integrity checks:
    - Strictly increasing timestamps
    - Expected 5-minute intervals (300,000 ms)
    - UTC timezone normalization
    - Duplicate timestamp detection & resolution
    - Missing candle identification & gap reporting
    - OHLC logic consistency ($\text{High} \ge \max(\text{Open}, \text{Close})$, $\text{Low} \le \min(\text{Open}, \text{Close})$)
    - Positive OHLC values ($\text{Open}, \text{High}, \text{Low}, \text{Close} > 0$)
    - Non-negative volume ($\text{Volume} \ge 0$)
    - NaN / Inf value detection
    - Schema type validation
    - Date coverage & time span verification
    - Total row count validation
    - Dataset provenance tracking (exchange source, endpoint, timestamp acquired)
    - File hash / checksum verification (SHA-256)
  - Output dataset metadata summary report.
- **Deliverables:** `src/crypto_analyzer/data/`, `scripts/download_data.py`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_data_pipeline.py`
- **Status:** **COMPLETE**

---

## Phase 3: Feature Engineering

- **Objective:** Compute returns, technical indicators, and volatility metrics with zero future look-ahead bias.
- **Tasks:**
  - Implement return & log-return generators.
  - Implement technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR).
  - Implement volatility features (rolling standard deviation of returns, normalized ATR).
  - Validate temporal alignment and NaN handling.
- **Deliverables:** `src/crypto_analyzer/features/`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_features.py`
- **Status:** NOT_STARTED

---

## Phase 4: Target & Sequence Pipeline

- **Objective:** Transform feature dataframes into model-ready sequence tensors with strict chronological splitting and train-only scaling.
- **Tasks:**
  - Build sliding-window 3D tensor generator with historical context $W = 60$ candles (5 hours) and feature dimension $K$: `(samples, W=60, K)`.
  - Build primary forecast targets with forecast horizon $H = 12$ candles (1 hour ahead):
    - Percentage return regression target: $R(t, 12) = \left( \frac{\text{Close}[t+12] - \text{Close}[t]}{\text{Close}[t]} \right) \times 100$
    - Binary directional classification target: $D(t, 12) = 1 \text{ if } R(t, 12) > 0 \text{ else } 0$
  - Implement chronological split (70% train, 15% validation, 15% test).
  - Fit scalers ONLY on training data; transform validation and test sets without re-fitting.
- **Deliverables:** `src/crypto_analyzer/preprocessing/`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_preprocessing.py`
- **Status:** NOT_STARTED

---

## Phase 5: Baseline Models

- **Objective:** Implement and benchmark traditional machine learning baselines.
- **Tasks:**
  - Naive / Lag baseline model.
  - Linear / Logistic Regression.
  - Random Forest Classifier & Regressor.
  - XGBoost Classifier & Regressor.
  - Save performance benchmarks and prediction outputs.
- **Deliverables:** `src/crypto_analyzer/models/baselines.py`, baseline experiment configs.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_baselines.py`
- **Status:** NOT_STARTED

---

## Phase 6: Deep Learning Models

- **Objective:** Implement modern modular deep learning models (LSTM, GRU, 1D-CNN) and unified PyTorch training loops.
- **Tasks:**
  - Build PyTorch LSTM model class.
  - Build PyTorch GRU model class.
  - Build PyTorch 1D-CNN model class.
  - Implement unified Trainer class with early stopping, learning rate scheduler, and checkpointing.
- **Deliverables:** `src/crypto_analyzer/models/dl/`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_dl_models.py`
- **Status:** NOT_STARTED

---

## Phase 7: Feature Ablation Study

- **Objective:** Conduct controlled feature ablation experiments (A, B, C, D) across models.
- **Tasks:**
  - Run Exp A (Price only).
  - Run Exp B (Price + Volume).
  - Run Exp C (Price + Volume + Technical Indicators).
  - Run Exp D (Full feature set).
  - Save all run artifacts to `experiments/results/`.
- **Deliverables:** Feature ablation evaluation reports.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_ablation.py`
- **Status:** NOT_STARTED

---

## Phase 8: Market Regime Analysis

- **Objective:** Evaluate trained models under distinct mathematical market regimes.
- **Tasks:**
  - Implement mathematical regime classification (Bull, Bear, Sideways, High Volatility).
  - Slice test period metrics by market regime.
  - Compute performance breakdowns per regime.
- **Deliverables:** `src/crypto_analyzer/regimes/`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_regimes.py`
- **Status:** NOT_STARTED

---

## Phase 9: Experiment Analysis

- **Objective:** Synthesize research results into comparative tables, statistical significance tests, and diagnostic plots.
- **Tasks:**
  - Generate metric comparison tables (Accuracy, F1, Directional Accuracy, MAE, RMSE).
  - Plot ROC curves, confusion matrices, and prediction vs actual return scatter plots.
  - Perform paired statistical significance tests following `DECISIONS.md` (Decision 011), utilizing Wilcoxon signed-rank tests where appropriate to the paired evaluation design rather than blindly applying to every metric.
- **Deliverables:** `reports/figures/`, `reports/tables/`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_analysis.py`
- **Status:** NOT_STARTED

---

## Phase 10: Crypto Analyzer Application

- **Objective:** Build an interactive Streamlit research analyzer dashboard.
- **Tasks:**
  - Display interactive historical charts & indicators.
  - Render model prediction outputs (direction, return, risk metrics).
  - Include mandatory scientific disclaimers and non-investment warnings.
- **Deliverables:** `src/crypto_analyzer/app.py`, `scripts/run_app.py`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/test_app.py`
- **Status:** NOT_STARTED

---

## Phase 11: Final Validation

- **Objective:** Audit the entire repository for reproducibility, data leakage, code quality, and test coverage.
- **Tasks:**
  - Run complete end-to-end test suite.
  - Perform clean-environment installation and execution check.
  - Audit code against PEP 8 and docstring standards.
- **Deliverables:** Audit report and final code fixes.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/ --cov=src/crypto_analyzer`
- **Status:** NOT_STARTED

---

## Phase 12: Research Deliverables

- **Objective:** Produce final research paper structure, methodology report, and executive summary.
- **Tasks:**
  - Write research paper summary in `reports/research/`.
  - Document final conclusions, limitations, and future research directions.
- **Deliverables:** `reports/research/final_paper.md`.
- **Validation Command:** `D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/`
- **Status:** NOT_STARTED
