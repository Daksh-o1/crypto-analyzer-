# CryptoAnalyzer Task Tracker (TASKS.md)

## CURRENT (Phase 0 — Project Foundation)
- [x] Verify D: drive existence and disk space
- [x] Create project root directory `D:\CryptoAnalyzer`
- [x] Initialize Git repository
- [x] Create Python 3.11 virtual environment on D: (`D:\CryptoAnalyzer\.venv`)
- [x] Create directory hierarchy (`src/`, `docs/`, `data/`, `experiments/`, `models/`, `reports/`, `tests/`, `scripts/`, `notebooks/`)
- [x] Create `.gitignore` and `.gitkeep` placeholders
- [x] Configure `requirements.txt` and `pyproject.toml`
- [x] Install testing framework (`pytest`, `pytest-cov`, `pandas`, `numpy`, `scikit-learn`, `pyyaml`)
- [x] Write core documentation (`README.md`, `phases.md`, `UPDATE.md`, `SUMMARY.md`, `RESEARCH.md`, `DECISIONS.md`, `TASKS.md`, `docs/*`)
- [x] Create basic test suite (`tests/conftest.py`, `tests/test_environment.py`)
- [x] Execute validation test suite (`pytest`)
- [x] Mark Phase 0 COMPLETE and STOP for Human Approval

## NEXT (Phase 1 — Research Foundation — Awaiting Approval)
- [ ] Detail mathematical formulation for directional classification & percentage return regression
- [ ] Formulate experimental ablation matrix (Experiments A, B, C, D)
- [ ] Document regime classification equations (Slope + Volatility thresholds)
- [ ] Document threats to validity and reproducibility protocols in `docs/methodology.md`

## BACKLOG
- [ ] Phase 2: Implement Binance BTC/USDT 5-minute candle downloader
- [ ] Phase 2: Implement dataset integrity checker (timestamp gaps, duplicate detection)
- [ ] Phase 3: Implement indicator generator (SMA, EMA, RSI, MACD, Bollinger, ATR)
- [ ] Phase 3: Implement volatility feature calculator
- [ ] Phase 4: Implement 3D sliding window tensor sequence generator
- [ ] Phase 4: Implement train-only scaler transformer
- [ ] Phase 5: Implement Naive, Linear Regression, Random Forest, XGBoost baselines
- [ ] Phase 6: Implement PyTorch LSTM, GRU, and 1D-CNN architectures & trainer
- [ ] Phase 7: Execute feature ablation matrix
- [ ] Phase 8: Execute market regime breakdown evaluation
- [ ] Phase 9: Generate scientific figures, metric tables, and paired significance tests
- [ ] Phase 10: Build interactive Streamlit analyzer app
- [ ] Phase 11: Execute complete repository audit & clean environment test
- [ ] Phase 12: Write research paper deliverables

## BLOCKED
- None at present.

## COMPLETED
- [x] Phase 0 project foundation & environment setup (2026-09-16)
