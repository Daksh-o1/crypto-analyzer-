# CryptoAnalyzer Research Specification (RESEARCH.md)

## 1. Research Problem & Scientific Motivation

Cryptocurrency asset markets, particularly Bitcoin (BTC/USDT), exhibit non-stationarity, heavy-tailed return distributions, non-linear dependencies, and time-varying regime transitions. While numerous public code repositories and academic tutorials attempt short-term cryptocurrency price forecasting, the vast majority suffer from severe, fatal methodological flaws:

1. **Look-Ahead Bias in Data Normalization:** Fitting feature scalers (e.g., `MinMaxScaler` or `StandardScaler`) on the combined dataset prior to train/test partitioning, thereby leaking future distribution statistics into training features.
2. **Temporal Data Leakage via Random Cross-Validation:** Utilizing random train/test splits or standard k-fold cross-validation on time-series observations, destroying temporal structure and training models on future data to predict past data.
3. **Non-Stationary Scalar Price Targets:** Attempting to directly predict nominal future price levels ($P_{t+H}$), which suffer from drift, scale variance, and statistical non-stationarity across different market cycles.
4. **Lack of Controlled Baselines & Ablation Studies:** Omitting trivial lag baselines or linear models, making it impossible to ascertain whether complex deep-learning architectures yield genuine predictive edge or merely overfit.
5. **Unconditioned Evaluation Across Market Regimes:** Evaluating aggregate performance across entire test sets without distinguishing between Bullish trends, Bearish trends, Sideways consolidation, or High Volatility regimes.

This research project establishes a rigorous, reproducible, open scientific framework (**CryptoAnalyzer**) to systematically address these deficiencies. We perform controlled 1-hour ahead ($H=12$ candles = 60 minutes) directional classification and percentage return regression on 5-minute BTC/USDT market data over a 5-hour historical context window ($W=60$ candles), utilizing strict chronological partitioning, train-only feature normalization, a controlled feature ablation matrix, modular machine learning and deep learning models, and explicit mathematical market regime conditioning.

---

## 2. Research Questions

### Primary Research Question
> **"Does combining historical market information, technical indicators, and volatility features improve short-term cryptocurrency movement and return forecasting compared with using historical price information alone?"**

### Sub-Questions
- **SQ1 (Baseline Benchmarking):** What baseline directional accuracy and Mean Absolute Error (MAE) are achieved by simple Naive Lag and regularized Linear models on 1-hour target ($H=12$) BTC/USDT data?
- **SQ2 (Feature Ablation Impact):** Does the progressive inclusion of volume, momentum, trend, and volatility features produce a statistically significant improvement in directional accuracy and reduction in return MAE across model families?
- **SQ3 (Architectural Comparison):** How do modern deep-learning sequence models (LSTM, GRU, 1D-CNN) compare against non-linear tree ensembles (Random Forest, XGBoost) and linear baselines in short-term sequence forecasting?
- **SQ4 (Market Regime Conditioning):** How does predictive performance shift across mathematically defined market regimes (Bullish, Bearish, Sideways, High Volatility), and do certain model architectures exhibit regime-specific performance advantages?

---

## 3. Formal Scientific Hypotheses

- **H1 (Feature Efficacy Hypothesis):**
  *Adding technical indicators and market-derived volatility features produces a statistically significant improvement in 1-hour directional prediction accuracy ($DA$) and a reduction in return regression MAE compared to using raw historical price information alone.*
- **H2 (Architecture Efficacy Hypothesis):**
  *Deep sequence architectures (LSTM, GRU, 1D-CNN) capture short-term temporal dependencies more effectively than tree ensembles (Random Forest, XGBoost) and linear baselines, with recurrent models (LSTM/GRU) outperforming 1D-CNN in low-volatility regimes and 1D-CNN demonstrating superior feature extraction efficiency.*
- **H3 (Market Regime Conditioning Hypothesis):**
  *Model predictive accuracy is non-uniform across market regimes, exhibiting significantly higher directional accuracy during strong directional regimes (Bullish/Bearish) than during range-bound (Sideways) or extreme volatility regimes.*
- **H4 (Target Metric Divergence Hypothesis):**
  *Directional classification performance ($DA$) does not strictly correlate with numerical return regression error (MAE/RMSE); models optimizing squared return loss may produce sub-random directional accuracy when returns cluster near zero.*

---

## 4. Prediction Target & Horizon Formulations

Let $P_t$ denote the Close price of BTC/USDT at 5-minute candle index $t$. Predicting nominal price $P_{t+H}$ directly is statistically invalid due to price non-stationarity ($P_t \sim I(1)$). Therefore, targets are strictly transformed into scale-stable return and directional targets suitable for time-series modeling.

### 4.1 Temporal Resolution & Horizons
- **Granularity:** 5-minute OHLCV candles ($5m$).
- **Historical Context Lookback ($W$):** $W = 60$ candles (5 hours of historical context).
- **Primary Prediction Horizon ($H$):** $H = 12$ candles (60 minutes = 1 hour ahead).

### 4.2 1-Hour Future Percentage Return (Regression Target)
The relative percentage return over horizon $H=12$ steps ahead is defined as:
$$R(t, H) = \left( \frac{P_{t+H} - P_t}{P_t} \right) \times 100 = \left( \frac{P_{t+12} - P_t}{P_t} \right) \times 100$$
where $R(t, H) \in \mathbb{R}$ represents the continuous percentage change over the next 1-hour window.

### 4.3 1-Hour Future Directional Target (Classification Target)
The binary directional movement over horizon $H=12$ steps ahead is defined as:
$$D(t, H) = \begin{cases} 1 & \text{if } R(t, H) > 0 \quad (\text{UP}) \\ 0 & \text{otherwise} \quad (\text{DOWN / FLAT}) \end{cases}$$
where $D(t, H) \in \{0, 1\}$ represents the discrete price movement sign over the next 1-hour window.

---

## 5. Dataset Methodology & Governance

### 5.1 Asset Pair & Venue
- **Asset Pair:** `BTC/USDT`
- **Exchange Venue:** Binance Spot Market (highest spot volume and market depth)
- **Timeframe Resolution:** 5-minute OHLCV candles ($5m$)
- **Data Governance:** All raw and processed datasets reside strictly on `D:\CryptoAnalyzer\data\`. Raw candles are stored immutably in `data/raw/` with zero modification.

### 5.2 Chronological Partitioning Protocol
To eliminate time-series look-ahead bias and data leakage:
- **Training Set (70%):** Earliest chronological data used strictly for feature scaler fitting and model optimization.
- **Validation Set (15%):** Sequential intermediate partition used exclusively for hyperparameter tuning and early stopping.
- **Testing Set (15%):** Chronologically last partition representing unseen future market conditions for final out-of-sample evaluation.

*Note: No random shuffling, bootstrap resampling, or standard k-fold cross-validation is permitted on raw temporal sequences.*

### 5.3 Normalization & Leakage Prevention Protocol
All feature scaling transformations (e.g., `StandardScaler` or `MinMaxScaler`) are fitted **strictly** on the training partition:
$$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} X_i^{\text{train}}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (X_i^{\text{train}} - \mu_{\text{train}})^2}$$
The fitted parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are subsequently applied to scale validation and testing partitions without re-fitting.

---

## 6. Feature Engineering & Ablation Matrix

### 6.1 Feature Categories
1. **Raw Market Features:** `Open`, `High`, `Low`, `Close`, `Volume`.
2. **Return & Log-Return Features:**
   - 1-period simple return: $R_t = (P_t - P_{t-1}) / P_{t-1}$
   - 1-period log return: $r_t = \ln(P_t / P_{t-1})$
   - Multi-period lag returns: $R_{t-1}, R_{t-2}, R_{t-3}, R_{t-6}, R_{t-12}$
3. **Technical Indicators:**
   - *Trend / Moving Averages:* Simple Moving Averages ($\text{SMA}_{12}, \text{SMA}_{24}, \text{SMA}_{96}$), Exponential Moving Averages ($\text{EMA}_{12}, \text{EMA}_{26}$).
   - *Momentum:* Relative Strength Index ($\text{RSI}_{14}$), Moving Average Convergence Divergence ($\text{MACD}$, Signal Line, MACD Histogram).
   - *Volatility / Bands:* Bollinger Bands (Upper, Lower, Bandwidth, $\%B$), Average True Range ($\text{ATR}_{14}$).
4. **Volatility Features:**
   - Rolling standard deviation of 5-minute returns over 12, 24, and 96 periods ($\sigma_{12}, \sigma_{24}, \sigma_{96}$).
   - Normalized ATR ratio: $\text{ATR}_{14} / P_t$.

### 6.2 Experimental Feature Ablation Matrix
To systematically evaluate hypothesis **H1**, features are grouped into four controlled experimental feature sets:

| Feature Set ID | Name | Included Features | Target Hypothesis |
|---|---|---|---|
| `EXP_A_PRICE` | Price Only | `[Open, High, Low, Close]` | Baseline price dynamics |
| `EXP_B_PRICE_VOL` | Price + Volume | `EXP_A_PRICE` + `[Volume]` | Impact of trading volume |
| `EXP_C_TECH_IND` | Price + Volume + Tech Indicators | `EXP_B_PRICE_VOL` + `[SMA_12, SMA_24, SMA_96, EMA_12, EMA_26, RSI_14, MACD, MACD_Signal, MACD_Hist, BB_Upper, BB_Lower, BB_Width, BB_PctB, ATR_14]` | Impact of classical technical analysis |
| `EXP_D_FULL` | Full Feature Set | `EXP_C_TECH_IND` + `[Rolling_Vol_12, Rolling_Vol_24, Rolling_Vol_96, ATR_Ratio, Log_Returns]` | Complete market feature space |

---

## 7. Model Taxonomy & Comparison Matrix

To address sub-question **SQ3** and hypothesis **H2**, seven distinct model architectures spanning four model families are benchmarked under identical pipeline conditions:

| Model Family | Model Name | Architecture / Formulation Details | Model Role |
|---|---|---|---|
| **Baselines** | Naive Lag Baseline | $D(t, H) = D(t-H, H)$, $R(t, H) = R(t-H, H)$ | Minimal benchmark for zero-skill baseline |
| **Baselines** | Linear / Logistic Regression | Ridge L2 regularization for regression; Logistic regression for classification | Linear parametric benchmark |
| **Tree Ensembles** | Random Forest | 100–300 trees, constrained max depth (5–10), min samples split | Non-linear tree benchmark |
| **Tree Ensembles** | XGBoost | Gradient Boosted Decision Trees, shrinkage $\eta=0.05$, max depth 6 | Advanced non-linear tree benchmark |
| **Deep Learning** | Stacked LSTM | 2-layer LSTM (hidden dim 64), Dropout 0.2, Dense output layer | Recurrent temporal feature extractor |
| **Deep Learning** | Stacked GRU | 2-layer GRU (hidden dim 64), Dropout 0.2, Dense output layer | Lightweight recurrent sequence extractor |
| **Deep Learning** | 1D-CNN | 2 x 1D Conv layers (32/64 filters, kernel size 3), MaxPool, Dense layer | Spatial/local pattern sequence extractor |

### 7.1 Sequence Tensor Construction
For sequence models (LSTM, GRU, 1D-CNN), input features are transformed into 3D sequence tensors of shape:
$$X \in \mathbb{R}^{N \times W \times K}$$
where $N$ is the number of samples, $W = 60$ is the sliding lookback window length (60 candles = 5 hours of historical context), and $K$ is the number of active features in the experiment set.

---

## 8. Market Regime Definition Methodology

To address sub-question **SQ4** and hypothesis **H3**, the out-of-sample test period is segmented into four distinct mathematical market regimes using a rolling 24-hour lookback window ($W_{\text{regime}} = 288$ candles):

1. **Trend Slope ($S_t$):** Ordinary Least Squares (OLS) linear regression slope of Close prices over lookback window $W_{\text{regime}}$, normalized by mean price:
   $$S_t = \frac{\text{Slope}(P_{t-W_{\text{regime}}:t})}{\text{Mean}(P_{t-W_{\text{regime}}:t})}$$
2. **Rolling Return Volatility ($\sigma_t$):** Standard deviation of 5-minute relative returns over $W_{\text{regime}}$:
   $$\sigma_t = \sqrt{\frac{1}{W_{\text{regime}}} \sum_{k=0}^{W_{\text{regime}}-1} (R_{t-k} - \bar{R}_t)^2}$$

### Threshold Classification Logic

| Market Regime | Mathematical Condition | Financial Interpretation |
|---|---|---|
| **Bullish Regime** | $S_t > +\theta_{\text{slope}}$ AND $\sigma_t \le \theta_{\text{vol\_high}}$ | Upward price trend with moderate volatility |
| **Bearish Regime** | $S_t < -\theta_{\text{slope}}$ AND $\sigma_t \le \theta_{\text{vol\_high}}$ | Downward price trend with moderate volatility |
| **Sideways Regime** | $|S_t| \le \theta_{\text{slope}}$ AND $\sigma_t \le \theta_{\text{vol\_high}}$ | Range-bound / consolidation market |
| **High Volatility Regime** | $\sigma_t > \theta_{\text{vol\_high}}$ | Extreme market turbulence regardless of slope |

*Default parameters: $\theta_{\text{slope}} = 0.0005$ per 5-min step, $\theta_{\text{vol\_high}} = 90\text{th percentile of training return volatility}$.*

---

## 9. Evaluation Metrics

### 9.1 Classification Metrics (Directional Target $D(t, H)$)
- **Directional Accuracy ($DA$):**
  $$DA = \frac{\sum_{i=1}^N \mathbb{I}(\hat{D}_i = D_i)}{N}$$
- **Precision, Recall, & Macro F1-Score:** Standard binary metrics evaluating class-balanced predictive quality.
- **Confusion Matrix:** True Positive, False Positive, True Negative, False Negative breakdowns.

### 9.2 Regression Metrics (Percentage Return Target $R(t, H)$)
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{R}_i - R_i|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{R}_i - R_i)^2}$$
- **Mean Absolute Percentage Error (MAPE):**
  $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{\hat{R}_i - R_i}{R_i} \right|$$

---

## 10. Experimental Design & Statistical Significance Protocol

### 10.1 Complete 4 x 7 Controlled Matrix
The benchmark consists of evaluating 4 feature sets (`EXP_A` through `EXP_D`) across 7 model architectures, resulting in **28 controlled experiment runs**.

### 10.2 Statistical Significance Testing Protocol
To verify whether observed metric differences between feature sets or model architectures are statistically significant (testing **H1** and **H2**):
1. **Paired Wilcoxon Signed-Rank Test:** Applied to prediction error series $(\hat{R}_i - R_i)^2$ and directional hit series $\mathbb{I}(\hat{D}_i = D_i)$ across models.
2. **Significance Threshold:** Null hypothesis $H_0$ rejected at $\alpha = 0.05$ significance level ($p < 0.05$).

---

## 11. Threats to Validity & Safeguards

1. **Internal Validity (Look-Ahead & Data Leakage):**
   - *Safeguard:* Strict chronological splitting (70% train / 15% val / 15% test).
   - *Safeguard:* Feature scalers fit **only** on training data.
   - *Safeguard:* Target values $R(t, H)$ and $D(t, H)$ computed prior to sliding window sequence assembly to prevent alignment overlap.
2. **External Validity (Generalizability):**
   - *Safeguard:* Primary testing conducted on BTC/USDT spot data, the highest liquidity asset in cryptocurrency. Limitations regarding altcoin transferability are explicitly documented.
3. **Construct Validity (Metric Divergence):**
   - *Safeguard:* Dual evaluation using both classification ($DA$, F1) and regression (MAE, RMSE) metrics to catch sign flips near zero returns (**H4**).
4. **Overfitting & Noise Risk:**
   - *Safeguard:* Regularization (L2, Dropout=0.2, constrained tree depth) and early stopping based strictly on validation set loss.

---

## 12. Project Limitations

1. **Single Asset Scope:** Focus is restricted to BTC/USDT spot market candles; results may differ for lower liquidity altcoins or derivative contracts.
2. **Technical Feature Boundary:** Features are limited to OHLCV derivatives, moving averages, momentum indicators, and volatility metrics. High-frequency L2 order book order flow, funding rates, on-chain metrics, and sentiment data are outside current scope.
3. **Frictionless Market Assumption:** Theoretical evaluation assumes frictionless execution (ignoring exchange fees, slippage, and order execution latency). Predictions are intended strictly for scientific benchmarking and must **NOT** be interpreted as actionable financial signals.

---

## 13. Relationship to Reference Repository (`khuangaf/CryptocurrencyPrediction`)

The baseline reference repository (`khuangaf/CryptocurrencyPrediction`) provided early inspiration for deep learning application to crypto prices, but suffers from critical methodological and technical shortcomings. The table below outlines how **CryptoAnalyzer** modernizes and corrects these issues:

| Dimension | Reference Repository (`CryptocurrencyPrediction`) | Modernized CryptoAnalyzer Framework |
|---|---|---|
| **Python / Stack** | Obsolete Python 3.6, Keras 2.x, TensorFlow 1.x | Modern Python 3.11, PyTorch 2.x, Scikit-Learn 1.4+ |
| **Data Normalization** | Global `MinMaxScaler` fit across full dataset prior to splitting (Severe Look-Ahead Bias) | Strict train-only scaler fit (`fit` on train, `transform` on val/test) |
| **Data Partitioning** | Unspecified or random time-series split | Strict chronological 70% Train / 15% Val / 15% Test split |
| **Prediction Target** | Direct nominal scalar price $P_{t+H}$ (Non-stationary) | Scale-stable 1-hour percentage return $R(t, H=12)$ & binary direction $D(t, H=12)$ suitable for time-series modeling |
| **Feature Space** | Ad-hoc raw price/volume features without controlled ablation | Systematic 4-tier Feature Ablation Matrix (`EXP_A` through `EXP_D`) |
| **Model Taxonomy** | Basic single LSTM model | 7 Modular Models (Naive, Linear, RF, XGBoost, LSTM, GRU, 1D-CNN) |
| **Market Regimes** | Zero market regime conditioning | 4 Mathematical Regimes (Bullish, Bearish, Sideways, High Volatility) |
| **Testing Harness** | Zero unit tests or data integrity assertions | Automated `pytest` suite testing environment, data leakage, & spec alignment |
| **Code Structure** | Monolithic Jupyter notebooks | Fully modular Python package (`src/crypto_analyzer/`) |
