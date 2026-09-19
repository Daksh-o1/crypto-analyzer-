# Phase 9 Statistical Significance: Classification

**Method:** Two-sided paired Wilcoxon signed-rank test. $p$-values adjusted via Holm step-down correction.

**Note:** Statistical significance ($lpha = 0.05$) denotes that the median of differences is non-zero, not necessarily a large practical effect or 'best' model.

| Comparison | N (total) | N (non-zero) | Statistic (W) | Mean Diff | Raw $p$-value | Adj $p$-value | Significant |
|---|---|---|---|---|---|---|---|
| XGBoost: EXP_A vs EXP_B | 1282 | 267 | 376157.00 | 0.0476 | 4.6902e-03 | 4.2212e-02 | **YES** |
| XGBoost: EXP_A vs EXP_C | 1282 | 442 | 364495.50 | 0.0686 | 2.3912e-04 | 2.3912e-03 | **YES** |
| XGBoost: EXP_A vs EXP_D | 1282 | 406 | 355067.50 | 0.0811 | 9.3196e-06 | 1.0252e-04 | **YES** |
| LSTM: EXP_A vs EXP_B | 1282 | 0 | nan | 0.0000 | 1.0000e+00 | 1.0000e+00 | No |
| LSTM: EXP_A vs EXP_C | 1282 | 0 | nan | 0.0000 | 1.0000e+00 | 1.0000e+00 | No |
| LSTM: EXP_A vs EXP_D | 1282 | 0 | nan | 0.0000 | 1.0000e+00 | 1.0000e+00 | No |
| Naive vs Logistic (EXP_D) | 1282 | 673 | 375726.50 | -0.0585 | 5.7104e-03 | 4.5683e-02 | **YES** |
| Logistic vs Random Forest (EXP_D) | 1282 | 577 | 399770.50 | 0.0179 | 3.7277e-01 | 1.0000e+00 | No |
| Random Forest vs XGBoost (EXP_D) | 1282 | 383 | 386654.00 | 0.0351 | 5.1919e-02 | 3.6343e-01 | No |
| XGBoost vs LSTM (EXP_D) | 1282 | 619 | 110058.00 | 0.4828 | 1.0470e-121 | 1.2564e-120 | **YES** |
| LSTM vs GRU (EXP_D) | 1282 | 0 | nan | 0.0000 | 1.0000e+00 | 1.0000e+00 | No |
| GRU vs 1D-CNN (EXP_D) | 1282 | 0 | nan | 0.0000 | 1.0000e+00 | 1.0000e+00 | No |