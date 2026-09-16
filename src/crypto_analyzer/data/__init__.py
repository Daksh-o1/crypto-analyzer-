"""
Data pipeline package for CryptoAnalyzer.
Provides reproducible data acquisition, raw storage, and validation for BTC/USDT.
"""

from crypto_analyzer.data.downloader import BinanceDownloader
from crypto_analyzer.data.metadata import DatasetMetadata, ValidationReport, calculate_sha256
from crypto_analyzer.data.validator import DataValidator

__all__ = [
    "BinanceDownloader",
    "DataValidator",
    "DatasetMetadata",
    "ValidationReport",
    "calculate_sha256",
]
