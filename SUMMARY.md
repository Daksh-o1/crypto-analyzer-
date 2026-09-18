# CryptoAnalyzer Project Summary (SUMMARY.md)

- **Current Phase:** Phase 8 (Market Regime Analysis) — **COMPLETE**
- **Overall Progress:** 9 / 13 Phases Complete (Phases 0–8 COMPLETE; awaiting human approval for Phase 9)
- **Project Root:** `D:\CryptoAnalyzer`
- **Environment:** Python 3.11.15 (`D:\CryptoAnalyzer\.venv`)
- **Git Branch:** `main` (Initialized)

---

## Current Status Overview

### Working Infrastructure
- **Directory Structure:** All project directories (`src/`, `docs/`, `data/`, `experiments/`, `models/`, `reports/`, `tests/`, `scripts/`, `notebooks/`) operational on D: drive.
- **Virtual Environment:** Built at `D:\CryptoAnalyzer\.venv` using Python 3.11.15.
- **Dependencies Installed:** `pytest`, `pytest-cov`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `requests`, `pyyaml`, `xgboost`.
- **Testing Harness:** `pytest` configured via `pyproject.toml` — **197/197 tests passed, 100% pass rate** across all phases.
- **Research Specification:** Complete formal research definitions, RQ/SQ1-4, H1-H4, scale-stable targets ($R(t, H=12), D(t, H=12)$ 1-hour horizon over $W=60$ 5-hour context), 4-tier feature ablation matrix (`EXP_A` through `EXP_D`), 7-model taxonomy matrix, 4 mathematical market regimes, evaluation metrics, threats to validity.
- **Decision Log:** Decisions 001 through 019 recorded in `DECISIONS.md`.

### Datasets Status
- **Raw Data (`data/raw/btcusdt_5m_raw.csv`):** 8,640 rows, immutable. SHA-256: `a722751b2929acdcae48cb875be4c3d4904d17e99d45ea0b3f6d9a67a4cf1eb8`.
- **Processed Features (`data/processed/btcusdt_5m_features.csv`):** 8,640 rows × 25 columns (timestamp + 24 features); 8,544 usable (post warm-up). Max warm-up: 96 rows.
- **Sequence Tensors (`data/processed/tensors/*.npz`):** Generated for all 4 experiment configurations.
  - Row progression: 8,640 raw → 8,544 clean → 8,532 target-valid → 8,473 total sequences
  - EXP_A: (5909, 60, 4) | EXP_B: (5909, 60, 5) | EXP_C: (5909, 60, 19) | EXP_D: (5909, 60, 24) [train shapes]
  - Val and Test: (1282, 60, K) each | Chronological 70/15/15 split

### Baseline Performance Summary (Phase 5 complete)
- Benchmark results logged in `experiments/results/all_baselines_results.json`.
- Naive model yields best MAE baseline (~0.2053) for regression.
- Baseline classification performance varies across models and feature sets; XGBoost on EXP_A achieved 56.4% test accuracy.

### Models Status
- **Baselines (Phase 5):** 5 models (Naive, Ridge, Logistic, RandomForest, XGBoost) fully implemented, tested, and evaluated across all 4 experiment feature sets.
- **Deep Learning (Phase 6):** LSTM, GRU, and 1D-CNN modular architectures implemented with PyTorch `DLTrainer`. Training matrix across 4 experiments and 2 tasks successfully completed (24 total configurations). Checkpoints and results artifacts preserved.

### Blockers
- None. Phase 8 is verified and complete. Ready for Phase 9 (Experiment Analysis) upon approval.

---

## Immediate Next Step
- Await human approval to proceed to **Phase 9 (Experiment Analysis)**.

