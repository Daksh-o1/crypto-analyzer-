"""
Command-line script for downloading, validating, and persisting raw BTC/USDT 5-minute candles.
Usage:
    D:\\CryptoAnalyzer\\.venv\\Scripts\\python.exe scripts/download_data.py --days 30
    D:\\CryptoAnalyzer\\.venv\\Scripts\\python.exe scripts/download_data.py --mock --days 30
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Ensure src is in python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from crypto_analyzer.data import (
    BinanceDownloader,
    DataValidator,
    DatasetMetadata,
    ValidationReport,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and validate historical 5-minute BTC/USDT candles from Binance."
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Asset trading pair symbol.")
    parser.add_argument("--interval", type=str, default="5m", help="Candle timeframe interval.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/CryptoAnalyzer/data/raw",
        help="Directory path to store raw dataset.",
    )
    parser.add_argument(
        "--filename", type=str, default="btcusdt_5m_raw.csv", help="Output raw CSV filename."
    )
    parser.add_argument("--days", type=int, default=30, help="Number of historical days to fetch.")
    parser.add_argument(
        "--start-date", type=str, default=None, help="Start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="End date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic valid mock data (for offline testing/demonstrations).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CryptoAnalyzer Data Pipeline (Phase 2) ===")
    print(f"Symbol: {args.symbol} | Timeframe: {args.interval}")
    print(f"Output Directory: {output_dir}")

    downloader = BinanceDownloader(output_dir=output_dir)

    # Determine millisecond time bounds
    now_utc = datetime.now(timezone.utc)
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = now_utc

    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=args.days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Time Window: {start_dt.isoformat()} to {end_dt.isoformat()}")

    if args.mock:
        print("Mode: MOCK DATA GENERATION")
        expected_candles = int((end_ms - start_ms) // 300_000)
        df = downloader.generate_mock_klines(
            num_candles=max(expected_candles, 100), start_time_ms=start_ms
        )
    else:
        print("Mode: LIVE BINANCE API DOWNLOAD")
        try:
            df = downloader.fetch_historical_data(
                symbol=args.symbol,
                interval=args.interval,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            )
        except Exception as e:
            print(f"ERROR: Failed to fetch data from Binance API: {e}")
            print("Falling back to synthetic mock data generation for offline verification...")
            expected_candles = int((end_ms - start_ms) // 300_000)
            df = downloader.generate_mock_klines(
                num_candles=max(expected_candles, 100), start_time_ms=start_ms
            )

    print(f"Downloaded/Generated {len(df)} rows.")

    # Save raw CSV
    csv_path, metadata = downloader.save_raw_csv(
        df=df, filename=args.filename, symbol=args.symbol, interval=args.interval
    )
    print(f"Saved raw CSV to: {csv_path}")

    # Validate dataset
    validator = DataValidator()
    report = validator.validate(df, filepath=csv_path, expected_symbol=args.symbol)

    # Save metadata & report
    metadata_path = output_dir / "dataset_metadata.json"
    report_path = output_dir / "validation_report.json"
    metadata.save_json(metadata_path)
    report.save_json(report_path)

    print("\n--- Validation Summary ---")
    print(f"Validation Status: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Total Rows: {report.total_rows}")
    print(f"Date Coverage: {report.start_timestamp} -> {report.end_timestamp}")
    print(f"Missing Candles: {report.missing_candles}")
    print(f"Duplicate Rows: {report.duplicate_rows}")
    print(f"Invalid OHLC Rows: {report.invalid_ohlc_rows}")
    print(f"Negative Volume Rows: {report.negative_volume_rows}")
    print(f"NaN / Inf Rows: {report.nan_inf_rows}")
    print(f"SHA-256 Checksum: {metadata.sha256}")
    print(f"Metadata Saved: {metadata_path}")
    print(f"Validation Report Saved: {report_path}")

    if not report.passed:
        print("\nValidation Errors Encountered:")
        for err in report.errors:
            print(f" - {err}")
        sys.exit(1)

    print("\nPhase 2 Data Pipeline Execution Complete.")


if __name__ == "__main__":
    main()
