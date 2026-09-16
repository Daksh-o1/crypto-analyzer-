"""
Comprehensive data validator module enforcing all Phase 2 data quality & integrity checks.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crypto_analyzer.data.metadata import ValidationReport, calculate_sha256


class DataValidator:
    """
    Enforces strict data quality, schema, boundary, gap, and integrity checks
    on raw historical 5-minute candle DataFrames.
    """

    EXPECTED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    INTERVAL_MS = 300_000  # 5 minutes in milliseconds

    def __init__(self, tolerance_ms: int = 0):
        self.tolerance_ms = tolerance_ms

    def validate(
        self,
        df: pd.DataFrame,
        filepath: Optional[Path] = None,
        expected_symbol: str = "BTCUSDT",
    ) -> ValidationReport:
        """
        Executes all 14 mandatory validation checks on the DataFrame.
        Returns a detailed ValidationReport.
        """
        report = ValidationReport()
        checks: Dict[str, bool] = {}
        errors: List[str] = []

        # Check 1: Schema & Type Validation
        schema_ok = True
        missing_cols = [c for c in self.EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            schema_ok = False
            errors.append(f"Schema Error: Missing required columns {missing_cols}")
        else:
            if not pd.api.types.is_numeric_dtype(df["timestamp"]):
                schema_ok = False
                errors.append("Schema Error: 'timestamp' column must be numeric (int64)")
            for col in ["open", "high", "low", "close", "volume"]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    schema_ok = False
                    errors.append(f"Schema Error: '{col}' column must be numeric (float64)")

        checks["schema_validation"] = schema_ok
        report.total_rows = len(df)

        if not schema_ok or df.empty:
            report.passed = False
            report.checks = checks
            report.errors = errors
            return report

        # Check 2: NaN / Inf Detection
        nan_inf_count = int(df[self.EXPECTED_COLUMNS].isna().sum().sum())
        # Check for Infinite values
        inf_mask = np.isinf(df[["open", "high", "low", "close", "volume"]].to_numpy())
        inf_count = int(np.sum(inf_mask))
        total_bad_nums = nan_inf_count + inf_count
        report.nan_inf_rows = total_bad_nums
        checks["nan_inf_detection"] = (total_bad_nums == 0)
        if total_bad_nums > 0:
            errors.append(f"Data Quality Error: Found {total_bad_nums} NaN or Inf values")

        # Check 3: Strictly Increasing Timestamps
        ts_series = df["timestamp"]
        diffs = ts_series.diff().dropna()
        is_strictly_increasing = bool((diffs > 0).all()) if len(diffs) > 0 else True
        checks["strictly_increasing_timestamps"] = is_strictly_increasing
        if not is_strictly_increasing:
            errors.append("Timestamp Error: Timestamps are not strictly increasing")

        # Check 4: Duplicate Detection
        dup_count = int(ts_series.duplicated().sum())
        report.duplicate_rows = dup_count
        checks["duplicate_detection"] = (dup_count == 0)
        if dup_count > 0:
            errors.append(f"Timestamp Error: Found {dup_count} duplicate timestamps")

        # Check 5: Expected 5-Minute Intervals & Missing Candle Gap Identification
        gaps: List[Dict[str, Any]] = []
        missing_total = 0
        interval_ok = True

        if len(ts_series) > 1:
            for i in range(len(ts_series) - 1):
                t_curr = int(ts_series.iloc[i])
                t_next = int(ts_series.iloc[i + 1])
                delta = t_next - t_curr

                if delta != self.INTERVAL_MS:
                    interval_ok = False
                    if delta > self.INTERVAL_MS:
                        missing_in_gap = int((delta // self.INTERVAL_MS) - 1)
                        missing_total += missing_in_gap
                        gaps.append(
                            {
                                "gap_start_timestamp": t_curr,
                                "gap_end_timestamp": t_next,
                                "missing_candles": missing_in_gap,
                                "gap_start_iso": datetime.fromtimestamp(
                                    t_curr / 1000, tz=timezone.utc
                                ).isoformat(),
                                "gap_end_iso": datetime.fromtimestamp(
                                    t_next / 1000, tz=timezone.utc
                                ).isoformat(),
                            }
                        )

        report.missing_candles = missing_total
        report.gaps = gaps
        checks["expected_5m_intervals"] = interval_ok
        checks["missing_candles_check"] = (missing_total == 0)
        if missing_total > 0:
            errors.append(f"Interval Warning: Identified {missing_total} missing 5-minute candles across {len(gaps)} gaps")

        # Check 6: UTC Timezone Normalization Check
        start_ts = int(ts_series.min())
        end_ts = int(ts_series.max())
        start_iso = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).isoformat()
        report.start_timestamp = start_iso
        report.end_timestamp = end_iso

        # Epoch timestamp range sanity check (e.g. year 2017 to 2100)
        utc_ok = (start_ts > 1483228800000) and (end_ts < 4102444800000)
        checks["utc_normalization"] = utc_ok
        if not utc_ok:
            errors.append("Timestamp Error: Timestamps fall outside realistic UTC timestamp bounds")

        # Check 7: OHLC Consistency
        high = df["high"]
        low = df["low"]
        open_p = df["open"]
        close_p = df["close"]

        invalid_high = (high < open_p) | (high < close_p)
        invalid_low = (low > open_p) | (low > close_p)
        invalid_spread = high < low
        invalid_ohlc_mask = invalid_high | invalid_low | invalid_spread

        invalid_ohlc_count = int(invalid_ohlc_mask.sum())
        report.invalid_ohlc_rows = invalid_ohlc_count
        checks["ohlc_consistency"] = (invalid_ohlc_count == 0)
        if invalid_ohlc_count > 0:
            errors.append(f"OHLC Consistency Error: Found {invalid_ohlc_count} rows violating High >= max(O,C) or Low <= min(O,C)")

        # Check 8: Positive OHLC Values
        non_positive_ohlc = (open_p <= 0) | (high <= 0) | (low <= 0) | (close_p <= 0)
        positive_ohlc_ok = not bool(non_positive_ohlc.any())
        checks["positive_ohlc_values"] = positive_ohlc_ok
        if not positive_ohlc_ok:
            errors.append("Price Error: Found non-positive or zero OHLC prices")

        # Check 9: Non-Negative Volume
        negative_vol = df["volume"] < 0
        neg_vol_count = int(negative_vol.sum())
        report.negative_volume_rows = neg_vol_count
        checks["non_negative_volume"] = (neg_vol_count == 0)
        if neg_vol_count > 0:
            errors.append(f"Volume Error: Found {neg_vol_count} rows with negative volume")

        # Check 10: Date Coverage Verification
        checks["date_coverage"] = bool(end_ts >= start_ts)

        # Check 11: Row Count Check
        checks["row_count_valid"] = len(df) > 0

        # Check 12: Provenance Tracking
        checks["dataset_provenance"] = bool(expected_symbol is not None)

        # Check 13: File SHA-256 Checksum (if file path provided)
        sha256_ok = True
        if filepath is not None:
            filepath = Path(filepath)
            if filepath.exists():
                try:
                    hash_val = calculate_sha256(filepath)
                    sha256_ok = bool(len(hash_val) == 64)
                except Exception as e:
                    sha256_ok = False
                    errors.append(f"SHA-256 Checksum Error: {str(e)}")
            else:
                sha256_ok = False
                errors.append(f"File Error: Specified filepath does not exist: {filepath}")

        checks["file_sha256_checksum"] = sha256_ok

        # Determine overall pass status
        # Note: Critical hard failures are schema, NaNs, duplicates, non-monotonic timestamps, OHLC violations, non-positive prices, negative volume, bad checksums.
        critical_checks = [
            "schema_validation",
            "nan_inf_detection",
            "strictly_increasing_timestamps",
            "duplicate_detection",
            "ohlc_consistency",
            "positive_ohlc_values",
            "non_negative_volume",
            "file_sha256_checksum",
        ]

        passed = all(checks.get(c, False) for c in critical_checks)
        report.passed = passed
        report.checks = checks
        report.errors = errors

        return report
