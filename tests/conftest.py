"""
Pytest global fixtures for CryptoAnalyzer test suite.
"""

from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return absolute path to project root on D: drive."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return absolute path to data directory."""
    return project_root / "data"
