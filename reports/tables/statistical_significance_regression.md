# Phase 9 Statistical Significance: Regression

**Method:** Two-sided paired Wilcoxon signed-rank test. $p$-values adjusted via Holm step-down correction.

**Note:** Statistical significance ($lpha = 0.05$) denotes that the median of differences is non-zero, not necessarily a large practical effect or 'best' model.

| Comparison | N (total) | N (non-zero) | Statistic (W) | Mean Diff | Raw $p$-value | Adj $p$-value | Significant |
|---|---|---|---|---|---|---|---|
| XGBoost: EXP_A vs EXP_B | 1282 | 1282 | 349247.00 | 0.0123 | 2.9712e-06 | 5.9423e-06 | **YES** |
| XGBoost: EXP_A vs EXP_C | 1282 | 1282 | 382991.00 | -0.0142 | 3.3360e-02 | 3.3360e-02 | **YES** |
| XGBoost: EXP_A vs EXP_D | 1282 | 1282 | 310296.00 | -0.0710 | 2.7281e-14 | 1.6368e-13 | **YES** |
| LSTM: EXP_A vs EXP_B | 1282 | 1282 | 269969.00 | 0.0231 | 1.7030e-26 | 1.5327e-25 | **YES** |
| LSTM: EXP_A vs EXP_C | 1282 | 1282 | 338414.00 | -0.0050 | 4.0225e-08 | 1.2067e-07 | **YES** |
| LSTM: EXP_A vs EXP_D | 1282 | 1282 | 314586.00 | -0.0011 | 3.1683e-13 | 1.5841e-12 | **YES** |
| Naive vs Ridge (EXP_D) | 1282 | 1282 | 254493.00 | -0.0425 | 3.0972e-32 | 3.0972e-31 | **YES** |
| Ridge vs Random Forest (EXP_D) | 1282 | 1282 | 276473.00 | -0.0974 | 2.9407e-24 | 2.3525e-23 | **YES** |
| Random Forest vs XGBoost (EXP_D) | 1282 | 1282 | 304117.00 | 0.0317 | 6.6571e-16 | 4.6600e-15 | **YES** |
| XGBoost vs LSTM (EXP_D) | 1282 | 1282 | 250319.00 | 0.0844 | 6.9533e-34 | 7.6487e-33 | **YES** |
| LSTM vs GRU (EXP_D) | 1282 | 1282 | 325768.00 | 0.0183 | 1.1662e-10 | 4.6650e-10 | **YES** |
| GRU vs 1D-CNN (EXP_D) | 1282 | 1282 | 12175.00 | -0.8301 | 5.4939e-199 | 6.5927e-198 | **YES** |