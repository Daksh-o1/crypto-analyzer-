# Methodology & Experimental Design (docs/methodology.md)

## 1. Mathematical Target & Horizon Definitions

Let $P_t$ represent the closing price of BTC/USDT at 5-minute candle timestamp index $t$. Nominal price series $P_t$ is non-stationary ($P_t \sim I(1)$). To ensure targets are scale-stable and suitable for time-series modeling across different market cycles, prediction targets are defined as relative return and directional movement over a 1-hour horizon ($H = 12$ candles).

### 1.1 Temporal Resolution & Horizons
- **Candle Granularity:** 5-minute OHLCV candles ($5m$).
- **Historical Context Lookback ($W$):** $W = 60$ candles (5 hours of historical context).
- **Primary Prediction Horizon ($H$):** $H = 12$ candles (60 minutes = 1 hour ahead).

### 1.2 Percentage Return Target (Regression)
$$R(t, H) = \left( \frac{P_{t+H} - P_t}{P_t} \right) \times 100 = \left( \frac{P_{t+12} - P_t}{P_t} \right) \times 100$$
where $R(t, H) \in \mathbb{R}$ is the relative percentage return over the 1-hour window ahead.

### 1.3 Directional Movement Target (Classification)
$$D(t, H) = \begin{cases} 1 & \text{if } R(t, H) > 0 \quad (\text{UP}) \\ 0 & \text{otherwise} \quad (\text{DOWN / FLAT}) \end{cases}$$
where $D(t, H) \in \{0, 1\}$ is the binary classification indicator.

---

## 2. Temporal Partitioning & Data Leakage Prevention

### 2.1 Strict Chronological Splitting Protocol
Time-series datasets must preserve temporal ordering. The dataset is partitioned chronologically into three non-overlapping segments:
- **Training Set (70%):** $t \in [1, T_{\text{train}}]$
- **Validation Set (15%):** $t \in [T_{\text{train}}+1, T_{\text{val}}]$
- **Test Set (15%):** $t \in [T_{\text{val}}+1, T_{\text{total}}]$

No random shuffling, bootstrap sampling, or k-fold cross-validation across temporal boundaries is permitted.

### 2.2 Fit-Only-On-Train Feature Normalization
Feature scalers (`StandardScaler` or `MinMaxScaler`) are fitted exclusively on the training partition:
$$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} X_i^{\text{train}}$$
$$\sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (X_i^{\text{train}} - \mu_{\text{train}})^2}$$
Transformation of validation and test sets uses training statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$):
$$X^{\text{val}}_{\text{scaled}} = \frac{X^{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad X^{\text{test}}_{\text{scaled}} = \frac{X^{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

---

## 3. Feature Engineering Formulations

### 3.1 Return & Log-Return Features
- **Simple Return ($R_t$):** $R_t = (P_t - P_{t-1}) / P_{t-1}$
- **Log Return ($r_t$):** $r_t = \ln(P_t / P_{t-1})$
- **Lag Returns:** $R_{t-1}, R_{t-2}, R_{t-3}, R_{t-6}, R_{t-12}$

### 3.2 Technical Indicators
- **Simple Moving Average ($\text{SMA}_n$):**
  $$\text{SMA}_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$
- **Exponential Moving Average ($\text{EMA}_n$):**
  $$\text{EMA}_n(t) = \alpha P_t + (1 - \alpha) \text{EMA}_n(t-1), \quad \alpha = \frac{2}{n + 1}$$
- **Relative Strength Index ($\text{RSI}_n$):**
  $$\text{RSI}_n(t) = 100 - \frac{100}{1 + \frac{\text{EMA}_n(U_t)}{\text{EMA}_n(D_t)}}$$
  where $U_t = \max(P_t - P_{t-1}, 0)$ and $D_t = \max(P_{t-1} - P_t, 0)$.
- **MACD:**
  $$\text{MACD}(t) = \text{EMA}_{12}(t) - \text{EMA}_{26}(t)$$
  $$\text{Signal}(t) = \text{EMA}_9(\text{MACD}(t))$$
  $$\text{Histogram}(t) = \text{MACD}(t) - \text{Signal}(t)$$
- **Bollinger Bands:**
  $$\text{Upper}(t) = \text{SMA}_{20}(t) + 2 \cdot \sigma_{20}(t)$$
  $$\text{Lower}(t) = \text{SMA}_{20}(t) - 2 \cdot \sigma_{20}(t)$$
  $$\text{Width}(t) = \frac{\text{Upper}(t) - \text{Lower}(t)}{\text{SMA}_{20}(t)}$$
  $$\%B(t) = \frac{P_t - \text{Lower}(t)}{\text{Upper}(t) - \text{Lower}(t)}$$
- **Average True Range ($\text{ATR}_n$):**
  $$\text{TR}_t = \max(H_t - L_t, |H_t - P_{t-1}|, |L_t - P_{t-1}|)$$
  $$\text{ATR}_n(t) = \text{EMA}_n(\text{TR}_t)$$

### 3.3 Volatility Features
- **Rolling Return Volatility ($\sigma_m$):**
  $$\sigma_m(t) = \sqrt{\frac{1}{m} \sum_{i=0}^{m-1} (R_{t-i} - \bar{R}_{t,m})^2}, \quad m \in \{12, 24, 96\}$$
- **Normalized ATR Ratio:**
  $$\text{ATR\_Ratio}(t) = \frac{\text{ATR}_{14}(t)}{P_t}$$

---

## 4. 3D Sliding Window Sequence Tensor Construction

For sequential models (LSTM, GRU, 1D-CNN), input matrices are structured into sliding 3D window tensors:
$$\mathbf{X} \in \mathbb{R}^{N \times W \times K}$$
- $N$: Total number of sequence samples ($N = T - W - H + 1$).
- $W$: Lookback window size ($W = 60$ candles = 5 hours of historical context).
- $H$: Prediction horizon ($H = 12$ candles = 60 minutes = 1 hour ahead).
- $K$: Number of features in active feature set ($K \in \{4, 5, 19, 24\}$ for experiments A, B, C, D respectively).

For sequence step index $i$, input tensor matrix $X_i \in \mathbb{R}^{W \times K}$ contains feature vectors from $t - W + 1$ to $t$:
$$X_i = \begin{bmatrix} f_{t-W+1, 1} & \dots & f_{t-W+1, K} \\ \vdots & \ddots & \vdots \\ f_{t, 1} & \dots & f_{t, K} \end{bmatrix}$$
The associated target vector corresponds to step $t+H$ ($R(t, H)$ or $D(t, H)$).

---

## 5. Mathematical Market Regime Classification

The out-of-sample test period is segmented using rolling lookback windows ($W_{\text{regime}} = 288$ candles = 24 hours):

### 5.1 Trend Slope ($S_t$)
$$S_t = \frac{\sum_{k=0}^{W_{\text{regime}}-1} (k - \bar{k})(P_{t-W_{\text{regime}}+1+k} - \bar{P})}{\sum_{k=0}^{W_{\text{regime}}-1} (k - \bar{k})^2} \times \frac{1}{\bar{P}}$$

### 5.2 Volatility ($\sigma_t$)
$$\sigma_t = \sqrt{\frac{1}{W_{\text{regime}}} \sum_{k=0}^{W_{\text{regime}}-1} (R_{t-k} - \bar{R})^2}$$

### 5.3 Regime Rules
- **Bullish Regime:** $S_t > +\theta_{\text{slope}}$ AND $\sigma_t \le \theta_{\text{vol\_high}}$
- **Bearish Regime:** $S_t < -\theta_{\text{slope}}$ AND $\sigma_t \le \theta_{\text{vol\_high}}$
- **Sideways Regime:** $|S_t| \le \theta_{\text{slope}}$ AND $\sigma_t \le \theta_{\text{vol\_high}}$
- **High Volatility Regime:** $\sigma_t > \theta_{\text{vol\_high}}$

Default thresholds: $\theta_{\text{slope}} = 0.0005$, $\theta_{\text{vol\_high}} = \text{Percentile}_{90}(\sigma_{\text{train}})$.

---

## 6. Metric Formulas

### 6.1 Directional Accuracy ($DA$)
$$DA = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{D}_i = D_i)$$

### 6.2 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{R}_i - R_i|$$

### 6.3 Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{R}_i - R_i)^2}$$

### 6.4 Statistical Significance (Wilcoxon Signed-Rank Test)
$$W = \sum_{i=1}^{N_r} [\text{rank}(|d_i|) \cdot \text{sgn}(d_i)]$$
where $d_i = e_{i, \text{Model A}} - e_{i, \text{Model B}}$ is the paired difference in prediction error.
