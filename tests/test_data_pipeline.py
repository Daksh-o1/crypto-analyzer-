"""
Phase 2 Validation Test Suite: Data Pipeline & Raw Data Integrity
Tests BinanceDownloader, DataValidator, DatasetMetadata, and CLI downloader script.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import pandas as pd

from crypto_analyzer.data import (
    BinanceDownloader,
    DataValidator,
    DatasetMetadata,
    ValidationReport,
    calculate_sha256,
)


@pytest.fixture
def temp_raw_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_mock_df():
    return BinanceDownloader.generate_mock_klines(
        num_candles=100, start_time_ms=1704067200000, seed=42
    )


def test_downloader_initialization_and_mock(temp_raw_dir):
    downloader = BinanceDownloader(output_dir=temp_raw_dir)
    assert downloader.output_dir == temp_raw_dir

    df = downloader.generate_mock_klines(num_candles=50, seed=123)
    assert len(df) == 50
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["timestamp"].dtype == np.int64
    assert df["open"].dtype == np.float64
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["open"] > 0).all()
    assert (df["volume"] >= 0).all()


def test_downloader_raw_csv_persistence_and_sha256(temp_raw_dir, clean_mock_df):
    downloader = BinanceDownloader(output_dir=temp_raw_dir)
    csv_path, metadata = downloader.save_raw_csv(
        df=clean_mock_df, filename="test_btc_raw.csv", symbol="BTCUSDT", interval="5m"
    )

    assert csv_path.exists()
    assert metadata.row_count == len(clean_mock_df)
    assert metadata.symbol == "BTCUSDT"
    assert metadata.interval == "5m"
    assert len(metadata.sha256) == 64

    # Verify calculated SHA-256 matches
    calculated_hash = calculate_sha256(csv_path)
    assert calculated_hash == metadata.sha256


def test_validator_clean_dataset_passes(temp_raw_dir, clean_mock_df):
    downloader = BinanceDownloader(output_dir=temp_raw_dir)
    csv_path, metadata = downloader.save_raw_csv(df=clean_mock_df, filename="clean.csv")

    validator = DataValidator()
    report = validator.validate(clean_mock_df, filepath=csv_path, expected_symbol="BTCUSDT")

    assert report.passed is True
    assert report.total_rows == len(clean_mock_df)
    assert report.missing_candles == 0
    assert report.duplicate_rows == 0
    assert report.invalid_ohlc_rows == 0
    assert report.negative_volume_rows == 0
    assert report.nan_inf_rows == 0
    assert len(report.errors) == 0

    expected_checks = [
        "schema_validation",
        "strictly_increasing_timestamps",
        "expected_5m_intervals",
        "missing_candles_check",
        "utc_normalization",
        "duplicate_detection",
        "ohlc_consistency",
        "positive_ohlc_values",
        "non_negative_volume",
        "nan_inf_detection",
        "date_coverage",
        "row_count_valid",
        "dataset_provenance",
        "file_sha256_checksum",
    ]
    for check in expected_checks:
        assert report.checks.get(check) is True, f"Check '{check}' failed in clean dataset."


def test_validator_detects_schema_errors():
    validator = DataValidator()
    bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    report = validator.validate(bad_df)
    assert report.passed is False
    assert report.checks["schema_validation"] is False
    assert any("Missing required columns" in e for e in report.errors)


def test_validator_detects_out_of_order_timestamps(clean_mock_df):
    df = clean_mock_df.copy()
    # Swap timestamps at index 5 and 6
    tmp = df.loc[5, "timestamp"]
    df.loc[5, "timestamp"] = df.loc[6, "timestamp"]
    df.loc[6, "timestamp"] = tmp

    validator = DataValidator()
    report = validator.validate(df)
    assert report.passed is False
    assert report.checks["strictly_increasing_timestamps"] is False


def test_validator_detects_duplicate_timestamps(clean_mock_df):
    df = clean_mock_df.copy()
    df.loc[10, "timestamp"] = df.loc[9, "timestamp"]

    validator = DataValidator()
    report = validator.validate(df)
    assert report.passed is False
    assert report.duplicate_rows == 1
    assert report.checks["duplicate_detection"] is False


def test_validator_detects_gaps_and_missing_candles(clean_mock_df):
    df = clean_mock_df.copy()
    # Drop rows 15, 16, 17 creating a 20-min gap (3 missing 5-min candles)
    df = df.drop(index=[15, 16, 17]).reset_index(drop=True)

    validator = DataValidator()
    report = validator.validate(df)
    assert report.missing_candles == 3
    assert len(report.gaps) == 1
    assert report.gaps[0]["missing_candles"] == 3


def test_validator_detects_invalid_ohlc(clean_mock_df):
    df = clean_mock_df.copy()
    # Violate High >= Open/Close by setting High < Low
    df.loc[4, "high"] = df.loc[4, "low"] - 100.0

    validator = DataValidator()
    report = validator.validate(df)
    assert report.passed is False
    assert report.invalid_ohlc_rows == 1
    assert report.checks["ohlc_consistency"] is False


def test_validator_detects_non_positive_prices(clean_mock_df):
    df = clean_mock_df.copy()
    df.loc[2, "close"] = -10.0

    validator = DataValidator()
    report = validator.validate(df)
    assert report.passed is False
    assert report.checks["positive_ohlc_values"] is False


def test_validator_detects_negative_volume(clean_mock_df):
    df = clean_mock_df.copy()
    df.loc[7, "volume"] = -5.0

    validator = DataValidator()
    report = validator.validate(df)
    assert report.passed is False
    assert report.negative_volume_rows == 1
    assert report.checks["non_negative_volume"] is False


def test_validator_detects_nans_and_infs(clean_mock_df):
    df = clean_mock_df.copy()
    df.loc[3, "close"] = np.nan
    df.loc[8, "volume"] = np.inf

    validator = DataValidator()
    report = validator.validate(df)
    assert report.passed is False
    assert report.nan_inf_rows == 2
    assert report.checks["nan_inf_detection"] is False


def test_metadata_serialization_and_deserialization(temp_raw_dir):
    meta = DatasetMetadata(
        source="Binance Spot API",
        endpoint="https://api.binance.com/api/v3/klines",
        symbol="BTCUSDT",
        interval="5m",
        row_count=1000,
        sha256="a" * 64,
    )
    json_path = temp_raw_dir / "dataset_metadata.json"
    meta.save_json(json_path)

    assert json_path.exists()
    reloaded = DatasetMetadata.from_json(json_path)
    assert reloaded.symbol == "BTCUSDT"
    assert reloaded.row_count == 1000
    assert reloaded.sha256 == "a" * 64


def test_no_feature_engineering_or_model_imports_in_data_pipeline():
    """Audits Phase 2 code to verify zero leakage of features, targets, models, PyTorch, or Streamlit."""
    import inspect
    import crypto_analyzer.data.downloader as dl
    import crypto_analyzer.data.validator as val

    dl_source = inspect.getsource(dl)
    val_source = inspect.getsource(val)
    combined = dl_source + val_source

    forbidden_terms = ["torch", "streamlit", "sklearn", "rsi", "macd", "sma", "ema", "target", "sequence"]
    for term in forbidden_terms:
        assert f"import {term}" not in combined.lower(), f"Forbidden import found: {term}"
