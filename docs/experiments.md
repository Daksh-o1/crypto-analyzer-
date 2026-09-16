# Experiments Matrix & Configurations (docs/experiments.md)

## Feature Ablation Experiments

| Experiment ID | Feature Configuration | Input Features | Objective |
|---|---|---|---|
| `EXP_A_PRICE` | Price-Only Baseline | Open, High, Low, Close | Test predictive power of raw price history alone |
| `EXP_B_PRICE_VOL` | Price + Volume | Exp A + Volume | Evaluate marginal value of trade volume |
| `EXP_C_TECH_IND` | Price + Volume + Technical Indicators | Exp B + SMA, EMA, RSI, MACD, BB, ATR | Test value of traditional technical analysis |
| `EXP_D_FULL` | Full Feature Set | Exp C + Rolling Return Volatility, ATR Ratio | Test value of explicit volatility features |

## Evaluation Metrics Summary
- **Classification:** Directional Accuracy (DA), Precision, Recall, Macro F1.
- **Regression:** MAE, RMSE, MAPE.
- **Regime Evaluation:** Metric breakdown across Bullish, Bearish, Sideways, and High Volatility regimes.
