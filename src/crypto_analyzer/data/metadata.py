"""
Dataset metadata and validation report data structures.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetMetadata:
    """Stores dataset provenance and integrity metadata."""
    source: str = "Binance Spot API"
    endpoint: str = "https://api.binance.com/api/v3/klines"
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    acquisition_timestamp: str = ""
    date_coverage_start: str = ""
    date_coverage_end: str = ""
    row_count: int = 0
    sha256: str = ""

    def __post_init__(self):
        if not self.acquisition_timestamp:
            self.acquisition_timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: Path) -> "DatasetMetadata":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ValidationReport:
    """Stores results of comprehensive data quality validation."""
    passed: bool = False
    total_rows: int = 0
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    missing_candles: int = 0
    duplicate_rows: int = 0
    invalid_ohlc_rows: int = 0
    negative_volume_rows: int = 0
    nan_inf_rows: int = 0
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: Path) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath


def calculate_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
