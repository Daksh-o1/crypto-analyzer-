# Phase 8: Market Regime Analysis Report

## Methodology
- **Regime window (W_regime):** 288 candles (24.0 hours)
- **Theta slope threshold (documented default):** 0.0005
- **Theta slope threshold (used):** 0.00003994
  *(50th percentile of abs(slope) over training partition (rows 0..5979) — data-adaptive, train-only)*
- **Theta vol_high threshold:** 0.00189578
  *(derived as 90.0th percentile of training-partition rolling volatility — training partition only (rows 0..5979))*
- **Warm-up rows (regime):** 287 rows (rows lacking a full 288-candle lookback are labelled 'Warmup')

## Regime Classification Precedence
1. **High Volatility** — sigma_t > theta_vol_high (regardless of slope)
2. **Bullish** — S_t > +theta_slope AND sigma_t <= theta_vol_high
3. **Bearish** — S_t < -theta_slope AND sigma_t <= theta_vol_high
4. **Sideways** — |S_t| <= theta_slope AND sigma_t <= theta_vol_high

## Dataset Partitions
| Partition | Rows |
|---|---|
| Train | 5980 |
| Validation | 1282 |
| Test | 1282 |
| Total Feature Rows | 8544 |

## Test Partition Regime Counts & Proportions
| Regime | Count | Proportion |
|---|---|---|
| Bullish | 231 | 18.02% |
| Bearish | 299 | 23.32% |
| Sideways | 752 | 58.66% |
| High Volatility | 0 | 0.00% |
| Warmup | 0 | 0.00% |

*(Out of 1282 test observations; 1282 classified, 0 in warm-up.)*

## Descriptive Statistics by Regime (Test Partition)
| Regime | Count | Mean Close | Mean Return % | Std Return % | Mean Slope | Mean Volatility |
|---|---|---|---|---|---|---|
| Bullish | 231 | 78359.2108 | 0.000385 | 0.122947 | 0.00006747 | 0.00111590 |
| Bearish | 299 | 76155.1054 | -0.004317 | 0.162171 | -0.00010132 | 0.00142334 |
| Sideways | 752 | 77239.7203 | -0.000452 | 0.062575 | -0.00000431 | 0.00079273 |
| High Volatility | 0 | - | - | - | - | - |

## Observations
- Results are purely descriptive and observational.
- No causal claims are made about regime types and model performance.
- The theta_vol_high threshold was derived exclusively from training-partition data to prevent look-ahead leakage.
- Regime labels are produced independently of any model predictions.
- These regime labels are available for downstream use in Phase 9 (significance testing) and Phase 10 (visualization).

## Limitations
- Regime classification is based on a single rolling 24-hour window. Regime transitions may not be precisely aligned to exact structural breaks.
- Short sideways or high-volatility episodes that are subsumed into adjacent longer regimes may be underrepresented.
- The slope threshold theta_slope is a pre-specified constant rather than a data-adaptive threshold.
- Warm-up rows at the start of the test partition (if any) are excluded from regime-conditioned analysis.