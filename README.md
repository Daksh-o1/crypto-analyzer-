# Research-Oriented AI-based Cryptocurrency Market Analyzer

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Project Phase](https://img.shields.io/badge/phase-2%20Data%20Pipeline%20COMPLETE-brightgreen.svg)](phases.md)

## 1. Project Title & Overview

**CryptoAnalyzer** is a research-oriented machine learning and deep learning framework for empirical investigation into cryptocurrency market dynamics.

> **Disclaimer:** This software is purely for research and educational purposes. It does **NOT** provide financial advice, trading signals, guaranteed predictions, or automated trading functionality.

---

## 2. Problem Statement & Research Motivation

Cryptocurrency markets, particularly Bitcoin (BTC/USDT), exhibit high volatility, non-linear dynamics, noise, and time-varying regime shifts. While many commercial and web projects claim to deliver high-accuracy price predictions, they often suffer from severe methodological flaws:
- Look-ahead bias in data scaling or sequence creation
- Data leakage across train/validation/test splits (e.g., random shuffling of temporal data)
- Reliance on single scalar price predictions rather than directional/distributional metrics
- Lack of controlled baseline models to quantify marginal gains from complex architectures
- Absence of market regime breakdown (e.g., bull vs. bear vs. sideways vs. high-volatility)

This project addresses these gaps by establishing a reproducible scientific benchmark comparing traditional ML models (Linear Regression, Random Forest, XGBoost) and deep-learning architectures (LSTM, GRU, 1D-CNN) under controlled feature sets and explicit market regimes.

---

## 3. Primary Research Question

> **"Does combining historical market information, technical indicators, and volatility features improve short-term cryptocurrency movement and return forecasting compared with using historical price information alone?"**

---

## 4. Key Research Objectives

1. **Short-Term Target Formulation:** Evaluate future 1-hour directional classification (UP/DOWN sign of return $D(t, 12)$) alongside percentage return regression ($R(t, 12)$) over a 5-hour lookback context ($W=60$).
2. **Feature Ablation Study:** Methodically quantify the predictive value added by:
   - Price alone vs. Price + Volume
   - Price + Volume + Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
   - Price + Volume + Technical Indicators + Volatility Features (rolling return volatility, ATR ratio)
3. **Model Taxonomy Comparison:** Benchmark simple baselines (Naive lag, Linear Regression) against non-linear ensemble methods (Random Forest, XGBoost) and recurrent/convolutional deep networks (LSTM, GRU, 1D-CNN).
4. **Market Regime Conditioning:** Mathematically partition test performance into distinct regimes (Bullish, Bearish, Sideways, High Volatility) to test stability across market conditions.
5. **Strict Reproducibility & Leakage Prevention:** Ensure zero data leakage through chronological dataset splitting, fit-only-on-train normalization, and fixed seed configurations.

---

## 5. System Architecture & Project Structure

```
D:\CryptoAnalyzer
│
├── README.md                 # Primary project documentation
├── phases.md                 # Master phase plan & phase gate tracker
├── UPDATE.md                 # Chronological development log
├── SUMMARY.md                # Current project status & metrics
├── RESEARCH.md               # Research foundation & hypotheses
├── DECISIONS.md              # Architectural & methodological decision log
├── TASKS.md                  # Actionable task tracking
├── .gitignore                # Git exclusions
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package specification & pytest config
│
├── docs/                     # Detailed architectural & methodology docs
│   ├── architecture.md
│   ├── dataset.md
│   ├── methodology.md
│   ├── experiments.md
│   └── reproducibility.md
│
├── src/                      # Core modular Python library
│   └── crypto_analyzer/
│       ├── data/             # Downloader, validator, & raw data metadata
│       ├── features/         # Feature engineering & technical indicators
│       ├── preprocessing/    # Chronological splitters & scalers
│       ├── models/           # Baselines & Deep Learning architectures
│       ├── evaluation/       # Directional & regression metrics
│       ├── regimes/          # Mathematical regime classification
│       ├── visualization/    # Figures & diagnostic plots
│       └── utils/            # Config parser & logging utilities
│
├── scripts/                  # Command-line runners for pipelines & training
├── tests/                    # Unit, integration, & leakage test suite
│
├── data/                     # Local data storage (D: Drive only)
│   ├── raw/                  # Immutable raw BTC/USDT OHLCV datasets
│   ├── processed/
│   └── external/
│
├── experiments/              # Controlled experiment outputs
│   ├── configs/              # YAML configuration files
│   ├── runs/                 # Execution logs
│   └── results/              # Saved metrics (JSON/CSV)
│
├── models/                   # Model artifacts & checkpoints
│   ├── checkpoints/
│   └── final/
│
├── reports/                  # Generated research deliverables
│   ├── figures/
│   ├── tables/
│   └── research/
│
└── notebooks/                # Exploratory notebooks (non-production code)
```

---

## 6. Technology Stack

- **Language:** Python 3.11
- **Virtual Environment:** `uv` on D: Drive (`D:\CryptoAnalyzer\.venv`)
- **Data Manipulation:** `pandas`, `numpy`, `scipy`
- **Machine Learning & Baselines:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Configuration & Logging:** `pyyaml`, standard `logging`
- **Testing Infrastructure:** `pytest`, `pytest-cov`

---

## 7. Installation & Setup (D: Drive Only)

### Step 1: Clone / Navigate to Project Root on D: Drive
```powershell
cd D:\CryptoAnalyzer
```

### Step 2: Virtual Environment Setup
Ensure the Python virtual environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Run Validation Test Suite
```powershell
D:\CryptoAnalyzer\.venv\Scripts\python.exe -m pytest tests/
```

---

## 8. Current Project Status

- **Phase 0 (Project Foundation):** **COMPLETE**
- **Phase 1 (Research Foundation):** **COMPLETE**
- **Phase 2 (Data Pipeline):** **COMPLETE**
- **Phase 3 (Feature Engineering):** NOT_STARTED (Awaiting Human Approval)

Refer to [phases.md](phases.md) for full phase status breakdown and [UPDATE.md](UPDATE.md) for full execution log.

Refer to [phases.md](phases.md) for full phase status breakdown and [UPDATE.md](UPDATE.md) for full execution log.
