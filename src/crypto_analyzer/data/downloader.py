"""
Binance historical 5-minute candle data downloader module.
Fetches BTC/USDT OHLCV candles from Binance Spot API and persists raw data.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

from crypto_analyzer.data.metadata import DatasetMetadata, calculate_sha256


class BinanceDownloader:
    """
    Downloads historical OHLCV 5-minute candles from Binance Spot API.
    Raw datasets are stored immutably without altering market values.
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_INTERVAL = "5m"
    INTERVAL_MS = 300_000  # 5 minutes in milliseconds

    def __init__(self, output_dir: Optional[Union[str, Path]] = None, timeout: int = 10):
        self.output_dir = Path(output_dir) if output_dir else Path("D:/CryptoAnalyzer/data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_klines_chunk(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List]:
        """Fetches a single chunk of klines (max 1000 candles) from Binance API."""
        params = {
            "symbol": symbol.upper().replace("/", ""),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_historical_data(
        self,
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        max_candles: Optional[int] = None,
        sleep_interval: float = 0.1,
    ) -> pd.DataFrame:
        """
        Fetches historical klines paginating through start_time to end_time.
        Returns DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        all_klines: List[List] = []
        current_start = start_time_ms

        while True:
            chunk = self.fetch_klines_chunk(
                symbol=symbol,
                interval=interval,
                start_time_ms=current_start,
                end_time_ms=end_time_ms,
                limit=1000,
            )

            if not chunk:
                break

            all_klines.extend(chunk)

            if max_candles and len(all_klines) >= max_candles:
                all_klines = all_klines[:max_candles]
                break

            last_timestamp = chunk[-1][0]
            next_start = last_timestamp + self.INTERVAL_MS

            if end_time_ms and next_start >= end_time_ms:
                break

            if current_start == next_start:
                break

            current_start = next_start
            time.sleep(sleep_interval)

        df = self._parse_klines_to_dataframe(all_klines)
        return df

    @staticmethod
    def _parse_klines_to_dataframe(klines: List[List]) -> pd.DataFrame:
        """Parses Binance raw klines array into standard pandas DataFrame."""
        if not klines:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        records = []
        for item in klines:
            records.append(
                {
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )

        df = pd.DataFrame(records)
        df["timestamp"] = df["timestamp"].astype(np.int64)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)

        return df

    def save_raw_csv(
        self,
        df: pd.DataFrame,
        filename: str = "btcusdt_5m_raw.csv",
        symbol: str = DEFAULT_SYMBOL,
        interval: str = DEFAULT_INTERVAL,
    ) -> Tuple[Path, DatasetMetadata]:
        """
        Saves DataFrame as raw CSV and generates associated DatasetMetadata.
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)

        sha256_val = calculate_sha256(filepath)
        start_iso = ""
        end_iso = ""
        if not df.empty:
            start_ts = int(df["timestamp"].min())
            end_ts = int(df["timestamp"].max())
            start_iso = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).isoformat()

        metadata = DatasetMetadata(
            source="Binance Spot API",
            endpoint=self.BASE_URL,
            symbol=symbol,
            interval=interval,
            acquisition_timestamp=datetime.now(timezone.utc).isoformat(),
            date_coverage_start=start_iso,
            date_coverage_end=end_iso,
            row_count=len(df),
            sha256=sha256_val,
        )

        return filepath, metadata

    @staticmethod
    def generate_mock_klines(
        num_candles: int = 1000,
        start_time_ms: int = 1704067200000,  # 2024-01-01 00:00:00 UTC
        initial_price: float = 42000.0,
        volatility: float = 0.002,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates synthetic valid 5-minute OHLCV DataFrame for testing/offline runs.
        Ensures strict OHLC consistency, positive values, and 300,000 ms step intervals.
        """
        np.random.seed(seed)
        timestamps = [start_time_ms + i * 300_000 for i in range(num_candles)]

        opens = np.zeros(num_candles)
        highs = np.zeros(num_candles)
        lows = np.zeros(num_candles)
        closes = np.zeros(num_candles)
        volumes = np.zeros(num_candles)

        current_price = initial_price

        for i in range(num_candles):
            pct_change = np.random.normal(0, volatility)
            open_p = current_price
            close_p = open_p * (1.0 + pct_change)
            high_extra = abs(np.random.normal(0, volatility / 2.0)) * open_p
            low_extra = abs(np.random.normal(0, volatility / 2.0)) * open_p

            high_p = max(open_p, close_p) + high_extra
            low_p = min(open_p, close_p) - low_extra
            if low_p <= 0:
                low_p = min(open_p, close_p) * 0.999

            volume_val = abs(np.random.gamma(2.0, 10.0))

            opens[i] = round(open_p, 2)
            highs[i] = round(high_p, 2)
            lows[i] = round(low_p, 2)
            closes[i] = round(close_p, 2)
            volumes[i] = round(volume_val, 6)

            current_price = close_p

        df = pd.DataFrame(
            {
                "timestamp": np.array(timestamps, dtype=np.int64),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )
        return df
