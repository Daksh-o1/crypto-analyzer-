# CryptoAnalyzer Research Specification (RESEARCH.md)

## 1. Research Problem

Cryptocurrency markets display non-stationary temporal dynamics, extreme tail volatility, and frequent regime shifts. Most public financial machine learning tutorials claim high directional or return prediction accuracy but rely on flawed methodologies (e.g., random cross-validation on time-series, target leakage, look-ahead scaler fitting, or uncalibrated single-point price targets).

This research project systematically evaluates:
1. Whether short-term (5-minute) directional return forecasting can exceed naive benchmarks under strict temporal evaluation protocols.
2. The marginal explanatory power of adding technical indicators and volatility features over raw price and volume series.
3. The comparative efficacy of traditional statistical/tree-based models vs. recurrent and convolutional neural networks across distinct market regimes.

---

## 2. Research Questions

### Primary Research Question
> **"Does combining historical market information, technical indicators, and volatility features improve short-term cryptocurrency movement and return forecasting compared with using historical price information alone?"**

### Sub-Questions
1. **SQ1:** What is the baseline directional accuracy and MAE achievable by simple lag and linear models on 5-minute BTC/USDT data?
2. **SQ2:** Does the inclusion of volume, momentum, trend, and volatility features produce a statistically significant reduction in prediction error across model families?
3. **SQ3:** How do sequence architectures (LSTM, GRU, 1D-CNN) compare against non-linear tree ensembles (Random Forest, XGBoost) in terms of directional classification and return regression?
4. **SQ4:** Does model performance degrade or shift significantly across market regimes (Bullish, Bearish, Sideways, High Volatility)?

---

## 3. Formal Hypotheses

- **H1 (Feature Efficacy):**
  *Adding technical indicators and market-derived volatility features improves short-term BTC forecasting accuracy and reduces return MAE compared to using historical price information alone.*
- **H2 (Architecture Efficacy):**
  *LSTM, GRU, and 1D-CNN deep-learning sequence architectures capture short-term temporal dynamics differently, with recurrent models (LSTM/GRU) outperforming 1D-CNN in low-volatility regimes and 1D-CNN demonstrating superior feature extraction speed.*
- **H3 (Market Regime Conditioning):**
  *Model predictive power is non-uniform across market regimes, exhibiting higher directional accuracy during trending (Bull/Bear) regimes compared to range-bound (Sideways) or extreme volatility regimes.*
- **H4 (Target Metric Divergence):**
  *Directional prediction performance (UP/DOWN classification) does not strictly correlate with numerical return forecasting error (MAE/RMSE); a model with minimal MAE may still exhibit sub-random directional accuracy if sign predictions fail near zero returns.*

---

## 4. Prediction Target Definitions

Let $P_t$ denote the Close price of BTC/USDT at 5-minute candle timestamp $t$.

### 1-Step Future Percentage Return (Regression Target)
$$R_{t+1} = \frac{P_{t+1} - P_t}{P_t} \times 100$$

### 1-Step Directional Target (Classification Target)
$$D_{t+1} = \begin{cases} 1 & \text{if } R_{t+1} > 0 \quad (\text{UP}) \\ 0 & \text{if } R_{t+1} \le 0 \quad (\text{DOWN}) \end{cases}$$

---

## 5. Feature Engineering Categories

1. **Raw Market Features:** `Open`, `High`, `Low`, `Close`, `Volume`.
2. **Return Features:** 1-period simple return ($R_t$), 1-period log return $\ln(P_t / P_{t-1})$, multi-period lag returns ($R_{t-1}, R_{t-2}, \dots$).
3. **Technical Indicators:**
   - Trend / Moving Averages: Simple Moving Average (SMA_12, SMA_24, SMA_96), Exponential Moving Average (EMA_12, EMA_26).
   - Momentum: Relative Strength Index (RSI_14), Moving Average Convergence Divergence (MACD, Signal, Histogram).
   - Volatility / Range: Bollinger Bands (Upper, Lower, Width, %B), Average True Range (ATR_14).
4. **Volatility Features:**
   - Rolling standard deviation of 5-minute returns over 12, 24, and 96 periods.
   - Normalized ATR ratio ($\text{ATR}_{14} / P_t$).

---

## 6. Experimental Feature Sets (Ablation Matrix)

- **Experiment A (Price Only):** `[Open, High, Low, Close]`
- **Experiment B (Price + Volume):** `[Open, High, Low, Close, Volume]`
- **Experiment C (Price + Volume + Technical Indicators):** Exp B + `[SMA, EMA, RSI, MACD, Bollinger Bands, ATR]`
- **Experiment D (Full Feature Set):** Exp C + `[Rolling Volatility, ATR Ratio, Log Returns]`

---

## 7. Model Taxonomy & Comparison Matrix

| Model Category | Model Name | Architecture Details | Primary Role |
|---|---|---|---|
| **Baselines** | Naive Lag Baseline | $D_{t+1} = D_t$, $R_{t+1} = R_t$ | Minimal benchmark |
| **Baselines** | Linear / Logistic Regression | Ridge / L2 Regularization | Linear baseline |
| **Tree Ensembles** | Random Forest | 100-300 trees, max depth constrained | Non-linear tabular baseline |
| **Tree Ensembles** | XGBoost / Gradient Boosting | Gradient Boosted Trees | Advanced tree baseline |
| **Deep Learning** | LSTM | 2-layer Stacked LSTM, Dropout=0.2 | Recurrent temporal feature extractor |
| **Deep Learning** | GRU | 2-layer Stacked GRU, Dropout=0.2 | Lightweight recurrent extractor |
| **Deep Learning** | 1D-CNN | Multi-filter 1D Convolution + MaxPool | Local pattern temporal extractor |

---

## 8. Market Regime Definition Methodology

Regimes are defined mathematically using rolling lookback windows ($W = 288$ candles = 24 hours):

1. **Trend Slope ($S_t$):** Linear regression slope of close prices over window $W$, normalized by mean price.
2. **Rolling Volatility ($\sigma_t$):** Standard deviation of 5-minute returns over window $W$.

### Threshold Definitions
- **Bullish Regime:** $S_t > +\theta_{\text{slope}}$ and $\sigma_t \le \theta_{\text{vol\_high}}$
- **Bearish Regime:** $S_t < -\theta_{\text{slope}}$ and $\sigma_t \le \theta_{\text{vol\_high}}$
- **Sideways Regime:** $|S_t| \le \theta_{\text{slope}}$ and $\sigma_t \le \theta_{\text{vol\_high}}$
- **High Volatility Regime:** $\sigma_t > \theta_{\text{vol\_high}}$

---

## 9. Evaluation Metrics

### Classification Metrics (Directional Prediction)
- Directional Accuracy ($\text{DA} = \frac{\text{Correct Sign Predictions}}{\text{Total Predictions}}$)
- Precision, Recall, Macro F1-Score
- Confusion Matrix

### Regression Metrics (Return Prediction)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## 10. Threats to Validity & Leakage Safeguards

1. **Look-Ahead Bias:** Scalers (MinMax / Standard) are fitted **strictly** on the training partition and applied to validation/testing partitions.
2. **Temporal Data Leakage:** Chronological splitting (70% Train, 15% Val, 15% Test) is strictly enforced. No random k-fold cross-validation is used on raw time-series sequences.
3. **Target Leakage:** Target $R_{t+1}$ and $D_{t+1}$ are computed prior to feature scaling and aligned strictly such that input feature matrix at time $t$ uses information $\le t$.
