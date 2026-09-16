"""
Phase 0 environment and project structure validation tests.
"""

import sys
from pathlib import Path
import pytest
import crypto_analyzer


def test_python_version():
    """Verify python major and minor versions meet project requirements."""
    assert sys.version_info >= (3, 10), "Python 3.10+ is required."


def test_package_import():
    """Verify crypto_analyzer package can be imported and version is set."""
    assert hasattr(crypto_analyzer, "__version__")
    assert crypto_analyzer.__version__ == "0.1.0"


def test_project_drive_location(project_root: Path):
    """Verify project root resides on D: drive per Rule 1."""
    assert project_root.drive.upper() == "D:", f"Project root must be on D: drive, got {project_root.drive}"


def test_directory_structure_exists(project_root: Path):
    """Verify mandatory directories exist within project root."""
    required_dirs = [
        "docs",
        "src/crypto_analyzer/data",
        "src/crypto_analyzer/features",
        "src/crypto_analyzer/preprocessing",
        "src/crypto_analyzer/models",
        "src/crypto_analyzer/evaluation",
        "src/crypto_analyzer/regimes",
        "src/crypto_analyzer/visualization",
        "src/crypto_analyzer/utils",
        "scripts",
        "tests",
        "data/raw",
        "data/processed",
        "data/external",
        "experiments/configs",
        "experiments/runs",
        "experiments/results",
        "models/checkpoints",
        "models/final",
        "reports/figures",
        "reports/tables",
        "reports/research",
        "notebooks",
    ]
    for rel_path in required_dirs:
        dir_path = project_root / rel_path
        assert dir_path.is_dir(), f"Required directory missing: {rel_path}"


def test_mandatory_docs_exist(project_root: Path):
    """Verify mandatory root documentation files exist."""
    required_files = [
        "README.md",
        "phases.md",
        "UPDATE.md",
        "SUMMARY.md",
        "RESEARCH.md",
        "DECISIONS.md",
        "TASKS.md",
        ".gitignore",
        "requirements.txt",
        "pyproject.toml",
    ]
    for filename in required_files:
        filepath = project_root / filename
        assert filepath.is_file(), f"Required file missing: {filename}"
