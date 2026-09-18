# Phase 8 Regime-wise Model Performance: Classification

| Experiment | Model | Regime | Sample Count | Accuracy | Precision | Recall | F1 Macro | F1 Binary |
|---|---|---|---|---|---|---|---|---|
| EXP_A_PRICE | naive | Bullish | 231 | 0.5195 | 0.5089 | 0.5044 | 0.5192 | nan |
| EXP_A_PRICE | naive | Bearish | 299 | 0.4615 | 0.4258 | 0.4783 | 0.4613 | nan |
| EXP_A_PRICE | naive | Sideways | 752 | 0.4707 | 0.4670 | 0.4745 | 0.4707 | nan |
| EXP_A_PRICE | naive | High Volatility | 0 | — | — | — | — | — |
| EXP_A_PRICE | logistic | Bullish | 231 | 0.4762 | 0.4750 | 0.6726 | 0.4583 | nan |
| EXP_A_PRICE | logistic | Bearish | 299 | 0.5151 | 0.4842 | 0.7754 | 0.4947 | nan |
| EXP_A_PRICE | logistic | Sideways | 752 | 0.5040 | 0.5000 | 0.8901 | 0.4208 | nan |
| EXP_A_PRICE | logistic | High Volatility | 0 | — | — | — | — | — |
| EXP_A_PRICE | random_forest | Bullish | 231 | 0.5152 | 0.5031 | 0.7168 | 0.4977 | nan |
| EXP_A_PRICE | random_forest | Bearish | 299 | 0.4348 | 0.4448 | 0.9058 | 0.3263 | nan |
| EXP_A_PRICE | random_forest | Sideways | 752 | 0.5891 | 0.5816 | 0.6113 | 0.5890 | nan |
| EXP_A_PRICE | random_forest | High Volatility | 0 | — | — | — | — | — |
| EXP_A_PRICE | xgboost | Bullish | 231 | 0.5368 | 0.5188 | 0.7345 | 0.5210 | nan |
| EXP_A_PRICE | xgboost | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | nan |
| EXP_A_PRICE | xgboost | Sideways | 752 | 0.6130 | 0.5949 | 0.6890 | 0.6111 | nan |
| EXP_A_PRICE | xgboost | High Volatility | 0 | — | — | — | — | — |
| EXP_A_PRICE | LSTM | Bullish | 231 | 0.5195 | 0.5067 | 0.6726 | 0.5101 | 0.5779 |
| EXP_A_PRICE | LSTM | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_A_PRICE | LSTM | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_A_PRICE | LSTM | High Volatility | 0 | — | — | — | — | — |
| EXP_A_PRICE | GRU | Bullish | 231 | 0.4978 | 0.4906 | 0.6903 | 0.4815 | 0.5735 |
| EXP_A_PRICE | GRU | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_A_PRICE | GRU | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_A_PRICE | GRU | High Volatility | 0 | — | — | — | — | — |
| EXP_A_PRICE | 1D-CNN | Bullish | 231 | 0.4892 | 0.4892 | 1.0000 | 0.3285 | 0.6570 |
| EXP_A_PRICE | 1D-CNN | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_A_PRICE | 1D-CNN | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_A_PRICE | 1D-CNN | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | naive | Bullish | 231 | 0.5195 | 0.5089 | 0.5044 | 0.5192 | nan |
| EXP_B_PRICE_VOL | naive | Bearish | 299 | 0.4615 | 0.4258 | 0.4783 | 0.4613 | nan |
| EXP_B_PRICE_VOL | naive | Sideways | 752 | 0.4707 | 0.4670 | 0.4745 | 0.4707 | nan |
| EXP_B_PRICE_VOL | naive | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | logistic | Bullish | 231 | 0.4589 | 0.4577 | 0.5752 | 0.4530 | nan |
| EXP_B_PRICE_VOL | logistic | Bearish | 299 | 0.5217 | 0.4874 | 0.7029 | 0.5139 | nan |
| EXP_B_PRICE_VOL | logistic | Sideways | 752 | 0.5160 | 0.5095 | 0.6461 | 0.5083 | nan |
| EXP_B_PRICE_VOL | logistic | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | random_forest | Bullish | 231 | 0.5411 | 0.5217 | 0.7434 | 0.5247 | nan |
| EXP_B_PRICE_VOL | random_forest | Bearish | 299 | 0.4649 | 0.4623 | 0.9783 | 0.3378 | nan |
| EXP_B_PRICE_VOL | random_forest | Sideways | 752 | 0.5452 | 0.5398 | 0.5630 | 0.5451 | nan |
| EXP_B_PRICE_VOL | random_forest | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | xgboost | Bullish | 231 | 0.5022 | 0.4933 | 0.6549 | 0.4924 | nan |
| EXP_B_PRICE_VOL | xgboost | Bearish | 299 | 0.4548 | 0.4573 | 0.9710 | 0.3229 | nan |
| EXP_B_PRICE_VOL | xgboost | Sideways | 752 | 0.5452 | 0.5322 | 0.6863 | 0.5367 | nan |
| EXP_B_PRICE_VOL | xgboost | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | LSTM | Bullish | 231 | 0.5152 | 0.5024 | 0.9204 | 0.4306 | 0.6500 |
| EXP_B_PRICE_VOL | LSTM | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_B_PRICE_VOL | LSTM | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_B_PRICE_VOL | LSTM | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | GRU | Bullish | 231 | 0.5368 | 0.5149 | 0.9204 | 0.4662 | 0.6603 |
| EXP_B_PRICE_VOL | GRU | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_B_PRICE_VOL | GRU | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_B_PRICE_VOL | GRU | High Volatility | 0 | — | — | — | — | — |
| EXP_B_PRICE_VOL | 1D-CNN | Bullish | 231 | 0.4892 | 0.4892 | 1.0000 | 0.3285 | 0.6570 |
| EXP_B_PRICE_VOL | 1D-CNN | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_B_PRICE_VOL | 1D-CNN | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_B_PRICE_VOL | 1D-CNN | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | naive | Bullish | 231 | 0.5195 | 0.5089 | 0.5044 | 0.5192 | nan |
| EXP_C_TECH_IND | naive | Bearish | 299 | 0.4615 | 0.4258 | 0.4783 | 0.4613 | nan |
| EXP_C_TECH_IND | naive | Sideways | 752 | 0.4707 | 0.4670 | 0.4745 | 0.4707 | nan |
| EXP_C_TECH_IND | naive | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | logistic | Bullish | 231 | 0.5238 | 0.5185 | 0.3717 | 0.5113 | nan |
| EXP_C_TECH_IND | logistic | Bearish | 299 | 0.5251 | 0.4896 | 0.6812 | 0.5199 | nan |
| EXP_C_TECH_IND | logistic | Sideways | 752 | 0.5359 | 0.5274 | 0.6193 | 0.5330 | nan |
| EXP_C_TECH_IND | logistic | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | random_forest | Bullish | 231 | 0.6407 | 0.6389 | 0.6106 | 0.6400 | nan |
| EXP_C_TECH_IND | random_forest | Bearish | 299 | 0.4783 | 0.4679 | 0.9493 | 0.3801 | nan |
| EXP_C_TECH_IND | random_forest | Sideways | 752 | 0.4867 | 0.4821 | 0.4692 | 0.4865 | nan |
| EXP_C_TECH_IND | random_forest | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | xgboost | Bullish | 231 | 0.4719 | 0.4571 | 0.4248 | 0.4702 | nan |
| EXP_C_TECH_IND | xgboost | Bearish | 299 | 0.5284 | 0.4943 | 0.9420 | 0.4663 | nan |
| EXP_C_TECH_IND | xgboost | Sideways | 752 | 0.4894 | 0.4893 | 0.6729 | 0.4726 | nan |
| EXP_C_TECH_IND | xgboost | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | LSTM | Bullish | 231 | 0.4156 | 0.4127 | 0.4602 | 0.4149 | 0.4351 |
| EXP_C_TECH_IND | LSTM | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_C_TECH_IND | LSTM | Sideways | 752 | 0.5572 | 0.5318 | 0.8954 | 0.5027 | 0.6673 |
| EXP_C_TECH_IND | LSTM | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | GRU | Bullish | 231 | 0.4329 | 0.4318 | 0.5044 | 0.4308 | 0.4653 |
| EXP_C_TECH_IND | GRU | Bearish | 299 | 0.4883 | 0.4740 | 0.9928 | 0.3735 | 0.6417 |
| EXP_C_TECH_IND | GRU | Sideways | 752 | 0.5146 | 0.5058 | 0.9383 | 0.4129 | 0.6573 |
| EXP_C_TECH_IND | GRU | High Volatility | 0 | — | — | — | — | — |
| EXP_C_TECH_IND | 1D-CNN | Bullish | 231 | 0.4892 | 0.4892 | 1.0000 | 0.3285 | 0.6570 |
| EXP_C_TECH_IND | 1D-CNN | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_C_TECH_IND | 1D-CNN | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_C_TECH_IND | 1D-CNN | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | naive | Bullish | 231 | 0.5195 | 0.5089 | 0.5044 | 0.5192 | nan |
| EXP_D_FULL | naive | Bearish | 299 | 0.4615 | 0.4258 | 0.4783 | 0.4613 | nan |
| EXP_D_FULL | naive | Sideways | 752 | 0.4707 | 0.4670 | 0.4745 | 0.4707 | nan |
| EXP_D_FULL | naive | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | logistic | Bullish | 231 | 0.6017 | 0.6105 | 0.5133 | 0.5977 | nan |
| EXP_D_FULL | logistic | Bearish | 299 | 0.4916 | 0.4650 | 0.6739 | 0.4828 | nan |
| EXP_D_FULL | logistic | Sideways | 752 | 0.5332 | 0.5241 | 0.6408 | 0.5283 | nan |
| EXP_D_FULL | logistic | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | random_forest | Bullish | 231 | 0.5152 | 0.5032 | 0.6991 | 0.5009 | nan |
| EXP_D_FULL | random_forest | Bearish | 299 | 0.4849 | 0.4626 | 0.7174 | 0.4682 | nan |
| EXP_D_FULL | random_forest | Sideways | 752 | 0.5319 | 0.5296 | 0.5040 | 0.5314 | nan |
| EXP_D_FULL | random_forest | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | xgboost | Bullish | 231 | 0.3463 | 0.3492 | 0.3894 | 0.3455 | nan |
| EXP_D_FULL | xgboost | Bearish | 299 | 0.4548 | 0.4528 | 0.8696 | 0.3798 | nan |
| EXP_D_FULL | xgboost | Sideways | 752 | 0.5359 | 0.5243 | 0.6944 | 0.5248 | nan |
| EXP_D_FULL | xgboost | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | LSTM | Bullish | 231 | 0.3983 | 0.4156 | 0.5664 | 0.3833 | 0.4794 |
| EXP_D_FULL | LSTM | Bearish | 299 | 0.4783 | 0.4692 | 0.9928 | 0.3543 | 0.6372 |
| EXP_D_FULL | LSTM | Sideways | 752 | 0.5199 | 0.5086 | 0.9517 | 0.4146 | 0.6629 |
| EXP_D_FULL | LSTM | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | GRU | Bullish | 231 | 0.4416 | 0.4333 | 0.4602 | 0.4415 | 0.4464 |
| EXP_D_FULL | GRU | Bearish | 299 | 0.5151 | 0.4875 | 0.9928 | 0.4219 | 0.6539 |
| EXP_D_FULL | GRU | Sideways | 752 | 0.5266 | 0.5132 | 0.8847 | 0.4600 | 0.6496 |
| EXP_D_FULL | GRU | High Volatility | 0 | — | — | — | — | — |
| EXP_D_FULL | 1D-CNN | Bullish | 231 | 0.4892 | 0.4892 | 1.0000 | 0.3285 | 0.6570 |
| EXP_D_FULL | 1D-CNN | Bearish | 299 | 0.4615 | 0.4615 | 1.0000 | 0.3158 | 0.6316 |
| EXP_D_FULL | 1D-CNN | Sideways | 752 | 0.4960 | 0.4960 | 1.0000 | 0.3316 | 0.6631 |
| EXP_D_FULL | 1D-CNN | High Volatility | 0 | — | — | — | — | — |