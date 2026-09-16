# Dataset Specification (docs/dataset.md)

## Asset Pair & Source
- **Symbol:** `BTC/USDT`
- **Exchange Venue:** Binance Spot Market
- **Timeframe:** 5-minute candles (`5m`)
- **Format:** Immutable raw CSV files stored in `data/raw/`

## Raw OHLCV Schema
Each row in the raw dataset represents a 5-minute candle interval containing:

| Field Name | Type | Description |
|---|---|---|
| `timestamp` | Int64 / Datetime | UNIX timestamp in milliseconds (UTC) |
| `open` | Float64 | Opening price of the candle interval |
| `high` | Float64 | Highest traded price during the interval |
| `low` | Float64 | Lowest traded price during the interval |
| `close` | Float64 | Closing price of the candle interval |
| `volume` | Float64 | Total traded base asset volume during the interval |

## Data Quality Checks & Validation Rules
1. **Timestamp Continuity:** Verify that consecutive timestamps strictly equal $t_{i} + 300,000$ milliseconds (5 minutes). Log and flag any missing intervals.
2. **Duplicate Detection:** Ensure zero duplicate timestamps exist.
3. **OHLC Boundary Conditions:**
   - $\text{High} \ge \max(\text{Open}, \text{Close})$
   - $\text{Low} \le \min(\text{Open}, \text{Close})$
   - $\text{Open}, \text{High}, \text{Low}, \text{Close} > 0$
4. **Volume Integrity:** $\text{Volume} \ge 0$.
