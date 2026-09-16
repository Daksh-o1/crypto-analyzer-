# Architecture Specification (docs/architecture.md)

## System Overview

The **CryptoAnalyzer** framework is designed as a modular, decoupled research pipeline.

```
+-------------------------------------------------------------------+
|                           DATA LAYER                              |
|   Binance Data Downloader -> Raw CSV Storage -> Data Validator    |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        FEATURE ENGINEERING                        |
|   Returns | Moving Averages | RSI/MACD | Bollinger/ATR | Volatility|
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     PREPROCESSING & SEQUENCING                    |
|   Chronological Split (70/15/15) -> Train Scaler -> 3D Windowing  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        MODEL EXECUTION LAYER                      |
|   Baselines (Naive, Linear, RF, XGBoost) | DL (LSTM, GRU, 1D-CNN)   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       EVALUATION & REGIMES                        |
|   Directional Acc | MAE/RMSE | Regime Segmentation | Statistical  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        REPORTING & APP LAYER                      |
|   Saved Results (JSON/CSV) -> Research Figures -> Streamlit App   |
+-------------------------------------------------------------------+
```

## Modular Components

1. **`crypto_analyzer.data`**: Handles API interaction with exchange endpoints, caching, data validation, gap filling, and immutable raw file management.
2. **`crypto_analyzer.features`**: Pure transformation modules that compute technical indicators and volatility features without look-ahead bias.
3. **`crypto_analyzer.preprocessing`**: Manages temporal chronological splitting, fits normalization scalers on training partitions only, and converts pandas DataFrames into 3D sequence arrays $(N, T, F)$.
4. **`crypto_analyzer.models`**: Modular model implementations exposing unified `.fit()` and `.predict()` interfaces for both sklearn-style baselines and PyTorch deep learning models.
5. **`crypto_analyzer.evaluation`**: Classification and regression metric evaluation, loss calculation, confusion matrices, and paired significance testing.
6. **`crypto_analyzer.regimes`**: Mathematical regime identification and slicing routines.
7. **`crypto_analyzer.visualization`**: Standardized plotting functions for research publication figures.
