# Methodology & Experimental Design (docs/methodology.md)

## 1. Temporal Splitting Protocol
To eliminate time-series look-ahead bias and temporal leakage:
- **Training Set (70%):** Chronologically earliest data used for feature scaling fit and model weight updates.
- **Validation Set (15%):** Middle chronological segment used exclusively for hyperparameter tuning and early stopping.
- **Test Set (15%):** Final chronological segment representing strictly unseen future market data.

## 2. Feature Normalization & Scaling
- Feature scaling (StandardScaler or MinMaxScaler) is computed **solely** using training set statistics:
  $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} X_i^{\text{train}}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (X_i^{\text{train}} - \mu_{\text{train}})^2}$$
- Validation and testing sets are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.

## 3. Sliding Window Tensor Construction
For input lookback window length $W$ (e.g. $W=60$ candles = 5 hours) and feature dimension $K$:
- Input Tensor $X_t \in \mathbb{R}^{W \times K}$
- Output Target $Y_t$:
  - Regression Target $R_{t+1} \in \mathbb{R}$
  - Direction Target $D_{t+1} \in \{0, 1\}$
