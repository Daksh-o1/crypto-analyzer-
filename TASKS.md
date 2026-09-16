# CryptoAnalyzer Task Tracker (TASKS.md)

## CURRENT (Phase 2 — Data Pipeline — COMPLETE)
- [x] Implement Binance BTC/USDT 5-minute historical candle downloader module (`BinanceDownloader`)
- [x] Implement dataset integrity & boundary validator (`DataValidator`):
  - [x] Strictly increasing timestamps
  - [x] Expected 5-minute intervals (300,000 ms)
  - [x] UTC normalization
  - [x] Duplicate timestamp detection
  - [x] Missing candle detection
  - [x] OHLC logic consistency check ($\text{High} \ge \max(\text{Open}, \text{Close})$, $\text{Low} \le \min(\text{Open}, \text{Close})$)
  - [x] Positive OHLC values check ($\text{Open}, \text{High}, \text{Low}, \text{Close} > 0$)
  - [x] Non-negative volume check ($\text{Volume} \ge 0$)
  - [x] NaN / Inf value detection
  - [x] Schema validation
  - [x] Date coverage verification
  - [x] Row count validation
  - [x] Dataset provenance logging
  - [x] File hash / checksum verification (SHA-256)
- [x] Store raw immutable datasets in `data/raw/btcusdt_5m_raw.csv` (8,640 rows, SHA-256: `a722751b...`)
- [x] Output dataset metadata summary report (`dataset_metadata.json`, `validation_report.json`)
- [x] Create CLI script `scripts/download_data.py`
- [x] Implement unit test suite `tests/test_data_pipeline.py` (13 passed)
- [x] Record Decision 015 in `DECISIONS.md`
- [x] Mark Phase 2 COMPLETE and STOP for Human Approval for Phase 3

## NEXT (Phase 3 — Feature Engineering — Awaiting Approval)
- [ ] Implement return & log-return generators ($R_t, r_t$, lag returns)
- [ ] Implement technical indicators (SMA_12/24/96, EMA_12/26, RSI_14, MACD/Signal/Hist, Bollinger Upper/Lower/Width/%B, ATR_14)
- [ ] Implement volatility feature calculators (rolling std dev of 5m returns over 12/24/96 periods, normalized ATR ratio)
- [ ] Validate temporal alignment and NaN handling with zero future look-ahead bias

## BACKLOG
- [ ] Phase 4: Implement 3D sliding window tensor sequence generator ($W=60, H=12$) & train-only scaler transformer
- [ ] Phase 5: Implement Naive, Linear Regression, Random Forest, and XGBoost baselines
- [ ] Phase 6: Implement PyTorch LSTM, GRU, and 1D-CNN architectures & trainer
- [ ] Phase 7: Execute feature ablation matrix (28 experiment runs)
- [ ] Phase 8: Execute market regime breakdown evaluation
- [ ] Phase 9: Generate scientific figures, metric tables, and paired significance tests
- [ ] Phase 10: Build interactive Streamlit analyzer app
- [ ] Phase 11: Execute complete repository audit & clean environment test
- [ ] Phase 12: Write research paper deliverables

## BLOCKED
- None at present.

## COMPLETED
- [x] Phase 0 project foundation & environment setup (2026-09-16)
- [x] Phase 1 research foundation & scientific specification (2026-09-16)
- [x] Phase 2 BTC/USDT raw data pipeline & 14-point validation (2026-09-16)
