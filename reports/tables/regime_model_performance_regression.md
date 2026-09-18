# Phase 8 Regime-wise Model Performance: Regression

| Experiment | Model | Regime | Sample Count | MAE | RMSE | MAPE |
|---|---|---|---|---|---|---|
| EXP_A_PRICE | naive | Bullish | 231 | 0.2721 | 0.3391 | 106.54% |
| EXP_A_PRICE | naive | Bearish | 299 | 0.3153 | 0.4508 | 105.27% |
| EXP_A_PRICE | naive | Sideways | 752 | 0.1410 | 0.2199 | 107.78% |
| EXP_A_PRICE | naive | High Volatility | 0 | — | — | — |
| EXP_A_PRICE | ridge | Bullish | 231 | 0.2740 | 0.3369 | 132.28% |
| EXP_A_PRICE | ridge | Bearish | 299 | 0.3310 | 0.4724 | 147.14% |
| EXP_A_PRICE | ridge | Sideways | 752 | 0.1398 | 0.2188 | 153.25% |
| EXP_A_PRICE | ridge | High Volatility | 0 | — | — | — |
| EXP_A_PRICE | random_forest | Bullish | 231 | 0.2655 | 0.3227 | 157.51% |
| EXP_A_PRICE | random_forest | Bearish | 299 | 0.4170 | 0.5504 | 309.48% |
| EXP_A_PRICE | random_forest | Sideways | 752 | 0.1551 | 0.2322 | 253.88% |
| EXP_A_PRICE | random_forest | High Volatility | 0 | — | — | — |
| EXP_A_PRICE | xgboost | Bullish | 231 | 0.2602 | 0.3201 | 153.77% |
| EXP_A_PRICE | xgboost | Bearish | 299 | 0.4543 | 0.5851 | 425.85% |
| EXP_A_PRICE | xgboost | Sideways | 752 | 0.1529 | 0.2284 | 217.76% |
| EXP_A_PRICE | xgboost | High Volatility | 0 | — | — | — |
| EXP_A_PRICE | LSTM | Bullish | 231 | 0.2710 | 0.3334 | 119.56% |
| EXP_A_PRICE | LSTM | Bearish | 299 | 0.3578 | 0.4868 | 289.07% |
| EXP_A_PRICE | LSTM | Sideways | 752 | 0.1630 | 0.2340 | 348.44% |
| EXP_A_PRICE | LSTM | High Volatility | 0 | — | — | — |
| EXP_A_PRICE | GRU | Bullish | 231 | 0.2712 | 0.3335 | 120.47% |
| EXP_A_PRICE | GRU | Bearish | 299 | 0.3488 | 0.4798 | 266.07% |
| EXP_A_PRICE | GRU | Sideways | 752 | 0.1624 | 0.2338 | 342.44% |
| EXP_A_PRICE | GRU | High Volatility | 0 | — | — | — |
| EXP_A_PRICE | 1D-CNN | Bullish | 231 | 1.8844 | 2.0054 | 2633.25% |
| EXP_A_PRICE | 1D-CNN | Bearish | 299 | 4.9673 | 5.0494 | 7814.15% |
| EXP_A_PRICE | 1D-CNN | Sideways | 752 | 3.5574 | 3.5747 | 13510.30% |
| EXP_A_PRICE | 1D-CNN | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | naive | Bullish | 231 | 0.2721 | 0.3391 | 106.54% |
| EXP_B_PRICE_VOL | naive | Bearish | 299 | 0.3153 | 0.4508 | 105.27% |
| EXP_B_PRICE_VOL | naive | Sideways | 752 | 0.1410 | 0.2199 | 107.78% |
| EXP_B_PRICE_VOL | naive | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | ridge | Bullish | 231 | 0.2774 | 0.3408 | 135.72% |
| EXP_B_PRICE_VOL | ridge | Bearish | 299 | 0.3328 | 0.4769 | 172.38% |
| EXP_B_PRICE_VOL | ridge | Sideways | 752 | 0.1487 | 0.2257 | 192.94% |
| EXP_B_PRICE_VOL | ridge | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | random_forest | Bullish | 231 | 0.2680 | 0.3253 | 151.72% |
| EXP_B_PRICE_VOL | random_forest | Bearish | 299 | 0.4543 | 0.5958 | 394.65% |
| EXP_B_PRICE_VOL | random_forest | Sideways | 752 | 0.1514 | 0.2286 | 277.42% |
| EXP_B_PRICE_VOL | random_forest | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | xgboost | Bullish | 231 | 0.2680 | 0.3289 | 157.03% |
| EXP_B_PRICE_VOL | xgboost | Bearish | 299 | 0.3912 | 0.5391 | 327.80% |
| EXP_B_PRICE_VOL | xgboost | Sideways | 752 | 0.1546 | 0.2268 | 265.19% |
| EXP_B_PRICE_VOL | xgboost | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | LSTM | Bullish | 231 | 0.2706 | 0.3364 | 105.70% |
| EXP_B_PRICE_VOL | LSTM | Bearish | 299 | 0.3161 | 0.4528 | 136.05% |
| EXP_B_PRICE_VOL | LSTM | Sideways | 752 | 0.1404 | 0.2197 | 123.78% |
| EXP_B_PRICE_VOL | LSTM | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | GRU | Bullish | 231 | 0.2686 | 0.3338 | 116.03% |
| EXP_B_PRICE_VOL | GRU | Bearish | 299 | 0.3251 | 0.4669 | 175.10% |
| EXP_B_PRICE_VOL | GRU | Sideways | 752 | 0.1422 | 0.2203 | 141.07% |
| EXP_B_PRICE_VOL | GRU | High Volatility | 0 | — | — | — |
| EXP_B_PRICE_VOL | 1D-CNN | Bullish | 231 | 0.4773 | 0.5409 | 641.29% |
| EXP_B_PRICE_VOL | 1D-CNN | Bearish | 299 | 1.0761 | 1.1739 | 1625.66% |
| EXP_B_PRICE_VOL | 1D-CNN | Sideways | 752 | 0.7114 | 0.7370 | 2654.19% |
| EXP_B_PRICE_VOL | 1D-CNN | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | naive | Bullish | 231 | 0.2721 | 0.3391 | 106.54% |
| EXP_C_TECH_IND | naive | Bearish | 299 | 0.3153 | 0.4508 | 105.27% |
| EXP_C_TECH_IND | naive | Sideways | 752 | 0.1410 | 0.2199 | 107.78% |
| EXP_C_TECH_IND | naive | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | ridge | Bullish | 231 | 0.2665 | 0.3305 | 177.95% |
| EXP_C_TECH_IND | ridge | Bearish | 299 | 0.3558 | 0.4883 | 247.35% |
| EXP_C_TECH_IND | ridge | Sideways | 752 | 0.1783 | 0.2491 | 380.12% |
| EXP_C_TECH_IND | ridge | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | random_forest | Bullish | 231 | 0.2781 | 0.3378 | 209.66% |
| EXP_C_TECH_IND | random_forest | Bearish | 299 | 0.4092 | 0.5465 | 373.88% |
| EXP_C_TECH_IND | random_forest | Sideways | 752 | 0.3102 | 0.4437 | 977.88% |
| EXP_C_TECH_IND | random_forest | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | xgboost | Bullish | 231 | 0.2771 | 0.3330 | 191.05% |
| EXP_C_TECH_IND | xgboost | Bearish | 299 | 0.3772 | 0.5262 | 332.21% |
| EXP_C_TECH_IND | xgboost | Sideways | 752 | 0.2025 | 0.2896 | 409.82% |
| EXP_C_TECH_IND | xgboost | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | LSTM | Bullish | 231 | 0.3377 | 0.3999 | 255.09% |
| EXP_C_TECH_IND | LSTM | Bearish | 299 | 0.3602 | 0.5353 | 232.54% |
| EXP_C_TECH_IND | LSTM | Sideways | 752 | 0.1501 | 0.2421 | 165.98% |
| EXP_C_TECH_IND | LSTM | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | GRU | Bullish | 231 | 0.2914 | 0.3472 | 180.51% |
| EXP_C_TECH_IND | GRU | Bearish | 299 | 0.3754 | 0.5601 | 341.39% |
| EXP_C_TECH_IND | GRU | Sideways | 752 | 0.1817 | 0.2777 | 376.23% |
| EXP_C_TECH_IND | GRU | High Volatility | 0 | — | — | — |
| EXP_C_TECH_IND | 1D-CNN | Bullish | 231 | 0.9032 | 0.9684 | 1227.07% |
| EXP_C_TECH_IND | 1D-CNN | Bearish | 299 | 1.4857 | 1.5847 | 2397.16% |
| EXP_C_TECH_IND | 1D-CNN | Sideways | 752 | 1.4686 | 1.5122 | 5626.95% |
| EXP_C_TECH_IND | 1D-CNN | High Volatility | 0 | — | — | — |
| EXP_D_FULL | naive | Bullish | 231 | 0.2721 | 0.3391 | 106.54% |
| EXP_D_FULL | naive | Bearish | 299 | 0.3153 | 0.4508 | 105.27% |
| EXP_D_FULL | naive | Sideways | 752 | 0.1410 | 0.2199 | 107.78% |
| EXP_D_FULL | naive | High Volatility | 0 | — | — | — |
| EXP_D_FULL | ridge | Bullish | 231 | 0.2765 | 0.3329 | 166.36% |
| EXP_D_FULL | ridge | Bearish | 299 | 0.3811 | 0.5033 | 289.58% |
| EXP_D_FULL | ridge | Sideways | 752 | 0.1860 | 0.2561 | 409.41% |
| EXP_D_FULL | ridge | High Volatility | 0 | — | — | — |
| EXP_D_FULL | random_forest | Bullish | 231 | 0.2791 | 0.3424 | 175.22% |
| EXP_D_FULL | random_forest | Bearish | 299 | 0.4667 | 0.5903 | 368.26% |
| EXP_D_FULL | random_forest | Sideways | 752 | 0.3171 | 0.4508 | 891.97% |
| EXP_D_FULL | random_forest | High Volatility | 0 | — | — | — |
| EXP_D_FULL | xgboost | Bullish | 231 | 0.2974 | 0.3562 | 174.90% |
| EXP_D_FULL | xgboost | Bearish | 299 | 0.4143 | 0.5394 | 276.83% |
| EXP_D_FULL | xgboost | Sideways | 752 | 0.2783 | 0.4581 | 612.41% |
| EXP_D_FULL | xgboost | High Volatility | 0 | — | — | — |
| EXP_D_FULL | LSTM | Bullish | 231 | 0.2919 | 0.3488 | 167.73% |
| EXP_D_FULL | LSTM | Bearish | 299 | 0.3324 | 0.5097 | 149.36% |
| EXP_D_FULL | LSTM | Sideways | 752 | 0.1686 | 0.3059 | 228.22% |
| EXP_D_FULL | LSTM | High Volatility | 0 | — | — | — |
| EXP_D_FULL | GRU | Bullish | 231 | 0.2762 | 0.3346 | 126.26% |
| EXP_D_FULL | GRU | Bearish | 299 | 0.3244 | 0.4814 | 177.71% |
| EXP_D_FULL | GRU | Sideways | 752 | 0.1454 | 0.2321 | 174.41% |
| EXP_D_FULL | GRU | High Volatility | 0 | — | — | — |
| EXP_D_FULL | 1D-CNN | Bullish | 231 | 0.5761 | 0.6677 | 824.94% |
| EXP_D_FULL | 1D-CNN | Bearish | 299 | 1.1218 | 1.1967 | 1869.64% |
| EXP_D_FULL | 1D-CNN | Sideways | 752 | 1.1513 | 1.1805 | 4337.49% |
| EXP_D_FULL | 1D-CNN | High Volatility | 0 | — | — | — |