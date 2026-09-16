# Experiments Matrix & Configurations (docs/experiments.md)

## 1. Feature Ablation Experiments Matrix

Input resolution: 5-minute BTC/USDT OHLCV candles.  
Context lookback window: $W = 60$ candles (5 hours of historical context).  
Prediction horizon: $H = 12$ candles (60 minutes = 1 hour ahead).

| Experiment ID | Feature Configuration | Input Feature Count | Primary Objective |
|---|---|---|---|
| `EXP_A_PRICE` | Price-Only Baseline | 4 (`[Open, High, Low, Close]`) | Test predictive power of raw price history alone |
| `EXP_B_PRICE_VOL` | Price + Volume | 5 (`EXP_A_PRICE` + `[Volume]`) | Evaluate marginal value of trade volume |
| `EXP_C_TECH_IND` | Price + Volume + Technical Indicators | 19 (`EXP_B_PRICE_VOL` + SMA, EMA, RSI, MACD, BB, ATR) | Test value of classical technical analysis features |
| `EXP_D_FULL` | Full Feature Set | 24 (`EXP_C_TECH_IND` + Rolling Vol, ATR Ratio, Log Returns) | Test value of explicit market volatility features |

### 1.1 Explicit Feature Column Enumeration

#### `EXP_A_PRICE` (Exactly 4 Features)
1. `open` — Opening price of 5-min candle
2. `high` — Highest price of 5-min candle
3. `low` — Lowest price of 5-min candle
4. `close` — Closing price of 5-min candle

#### `EXP_B_PRICE_VOL` (Exactly 5 Features)
1. `open` — Opening price
2. `high` — Highest price
3. `low` — Lowest price
4. `close` — Closing price
5. `volume` — Base asset volume traded

#### `EXP_C_TECH_IND` (Exactly 19 Features)
1. `open` — Opening price
2. `high` — Highest price
3. `low` — Lowest price
4. `close` — Closing price
5. `volume` — Base asset volume traded
6. `sma_12` — Simple Moving Average (12 periods = 1 hour)
7. `sma_24` — Simple Moving Average (24 periods = 2 hours)
8. `sma_96` — Simple Moving Average (96 periods = 8 hours)
9. `ema_12` — Exponential Moving Average (12 periods)
10. `ema_26` — Exponential Moving Average (26 periods)
11. `rsi_14` — Relative Strength Index (14 periods)
12. `macd` — Moving Average Convergence Divergence Line ($\text{EMA}_{12} - \text{EMA}_{26}$)
13. `macd_signal` — MACD Signal Line ($\text{EMA}_9(\text{MACD})$)
14. `macd_hist` — MACD Histogram ($\text{MACD} - \text{Signal}$)
15. `bb_upper` — Bollinger Bands Upper Band ($\text{SMA}_{20} + 2\sigma_{20}$)
16. `bb_lower` — Bollinger Bands Lower Band ($\text{SMA}_{20} - 2\sigma_{20}$)
17. `bb_width` — Bollinger Bands Width ($(\text{Upper} - \text{Lower}) / \text{SMA}_{20}$)
18. `bb_pct_b` — Bollinger Bands $\%B$ ($(\text{Close} - \text{Lower}) / (\text{Upper} - \text{Lower})$)
19. `atr_14` — Average True Range (14 periods)

#### `EXP_D_FULL` (Exactly 24 Features)
1. `open` — Opening price
2. `high` — Highest price
3. `low` — Lowest price
4. `close` — Closing price
5. `volume` — Base asset volume traded
6. `sma_12` — Simple Moving Average (12 periods)
7. `sma_24` — Simple Moving Average (24 periods)
8. `sma_96` — Simple Moving Average (96 periods)
9. `ema_12` — Exponential Moving Average (12 periods)
10. `ema_26` — Exponential Moving Average (26 periods)
11. `rsi_14` — Relative Strength Index (14 periods)
12. `macd` — MACD Line
13. `macd_signal` — MACD Signal Line
14. `macd_hist` — MACD Histogram
15. `bb_upper` — Bollinger Bands Upper Band
16. `bb_lower` — Bollinger Bands Lower Band
17. `bb_width` — Bollinger Bands Width
18. `bb_pct_b` — Bollinger Bands $\%B$
19. `atr_14` — Average True Range (14 periods)
20. `rolling_vol_12` — Rolling return standard deviation (12 periods = 1 hour)
21. `rolling_vol_24` — Rolling return standard deviation (24 periods = 2 hours)
22. `rolling_vol_96` — Rolling return standard deviation (96 periods = 8 hours)
23. `atr_ratio` — Normalized ATR Ratio ($\text{ATR}_{14} / \text{Close}$)
24. `log_returns` — 1-period log return ($\ln(\text{Close}_t / \text{Close}_{t-1})$)

---

## 2. Model Taxonomy & Matrix (4 x 7 Controlled Experiments)

Each feature set (`EXP_A` through `EXP_D`) is evaluated across 7 model architectures for prediction horizon $H=12$, producing **28 controlled experiment runs**:

| Model Architecture ID | Model Family | Hyperparameters / Setup | Sequence Input Shape |
|---|---|---|---|
| `NAIVE_LAG` | Baseline | $D(t, H) = D(t-H, H), R(t, H) = R(t-H, H)$ | Non-sequential |
| `RIDGE_LOGISTIC` | Baseline | Ridge L2 regularized ($\alpha=1.0$) | Flattened / 2D tabular |
| `RANDOM_FOREST` | Tree Ensemble | $N=200$ trees, `max_depth=8`, `min_samples_split=10` | Flattened / 2D tabular |
| `XGBOOST` | Tree Ensemble | `n_estimators=200`, `learning_rate=0.05`, `max_depth=6` | Flattened / 2D tabular |
| `STACKED_LSTM` | Deep Learning | 2-layer LSTM (dim 64), Dropout 0.2, Adam ($\eta=10^{-3}$) | 3D tensor $(N, W=60, K)$ |
| `STACKED_GRU` | Deep Learning | 2-layer GRU (dim 64), Dropout 0.2, Adam ($\eta=10^{-3}$) | 3D tensor $(N, W=60, K)$ |
| `CNN_1D` | Deep Learning | 2 x 1D-Conv (32/64 filters, $k=3$), MaxPool, Dense | 3D tensor $(N, W=60, K)$ |

---

## 3. Execution Pipeline & Output Artifacts

- **Config Files:** `experiments/configs/{exp_id}_{model_id}.yaml`
- **Execution Logs:** `experiments/runs/{exp_id}_{model_id}.log`
- **Results Summary:** `experiments/results/metrics_summary.json` and `experiments/results/ablation_results.csv`
- **Model Checkpoints:** `models/checkpoints/{exp_id}_{model_id}.pt`

---

## 4. Evaluation & Statistical Significance

- **Classification:** Directional Accuracy ($DA$), Precision, Recall, Macro F1, Confusion Matrix.
- **Regression:** MAE, RMSE, MAPE.
- **Regime Evaluation:** Metric breakdown across Bullish, Bearish, Sideways, and High Volatility regimes.
- **Significance Test:** Non-parametric paired Wilcoxon signed-rank test ($\alpha = 0.05$).
