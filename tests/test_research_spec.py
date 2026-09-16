"""
Phase 1 Validation Test Suite: Research Foundation Specification
Verifies that RESEARCH.md, docs/methodology.md, and DECISIONS.md accurately reflect all Phase 1 deliverables,
including scale-stable target definitions and explicit prediction horizon H=12 (1 hour) over W=60 (5 hours) lookback.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


def test_research_md_exists_and_populated():
    research_file = PROJECT_ROOT / "RESEARCH.md"
    assert research_file.exists(), "RESEARCH.md does not exist."
    content = research_file.read_text(encoding="utf-8")
    assert len(content) > 3000, "RESEARCH.md content is incomplete or too short."

    # Check key mandatory sections and updated formulations
    required_sections = [
        "Research Problem",
        "Primary Research Question",
        "Formal Scientific Hypotheses",
        "H1 (Feature Efficacy Hypothesis)",
        "H2 (Architecture Efficacy Hypothesis)",
        "H3 (Market Regime Conditioning Hypothesis)",
        "H4 (Target Metric Divergence Hypothesis)",
        "Prediction Target & Horizon Formulations",
        "1-Hour Future Percentage Return",
        "1-Hour Future Directional Target",
        "Dataset Methodology",
        "Chronological Partitioning Protocol",
        "Experimental Feature Ablation Matrix",
        "EXP_A_PRICE",
        "EXP_B_PRICE_VOL",
        "EXP_C_TECH_IND",
        "EXP_D_FULL",
        "Model Taxonomy & Comparison Matrix",
        "Naive Lag Baseline",
        "Linear / Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Stacked LSTM",
        "Stacked GRU",
        "1D-CNN",
        "Market Regime Definition Methodology",
        "Bullish Regime",
        "Bearish Regime",
        "Sideways Regime",
        "High Volatility Regime",
        "Evaluation Metrics",
        "Directional Accuracy",
        "Mean Absolute Error",
        "Threats to Validity",
        "Project Limitations",
        "khuangaf/CryptocurrencyPrediction",
    ]

    for section in required_sections:
        assert section in content, f"Missing required section '{section}' in RESEARCH.md"

    # Check scale-stable target terminology and horizon H=12
    assert "scale-stable" in content.lower(), "RESEARCH.md must mention scale-stable targets."
    assert "H = 12" in content or "H=12" in content, "RESEARCH.md must specify horizon H=12."
    assert "W = 60" in content or "W=60" in content, "RESEARCH.md must specify lookback window W=60."


def test_methodology_doc_exists_and_populated():
    methodology_file = PROJECT_ROOT / "docs" / "methodology.md"
    assert methodology_file.exists(), "docs/methodology.md does not exist."
    content = methodology_file.read_text(encoding="utf-8")
    assert len(content) > 1500, "docs/methodology.md content is incomplete or too short."

    required_elements = [
        "Percentage Return Target",
        "Directional Movement Target",
        "Strict Chronological Splitting Protocol",
        "Fit-Only-On-Train Feature Normalization",
        "RSI",
        "MACD",
        "Bollinger Bands",
        "ATR",
        "3D Sliding Window Sequence Tensor Construction",
        "Market Regime Classification",
        "Wilcoxon Signed-Rank Test",
    ]

    for elem in required_elements:
        assert elem in content, f"Missing required element '{elem}' in docs/methodology.md"

    # Check scale-stable terminology and horizon parameters
    assert "scale-stable" in content.lower(), "docs/methodology.md must mention scale-stable targets."
    assert "H = 12" in content or "H=12" in content, "docs/methodology.md must specify horizon H=12."


def test_decisions_log_contains_phase_1_decisions():
    decisions_file = PROJECT_ROOT / "DECISIONS.md"
    assert decisions_file.exists(), "DECISIONS.md does not exist."
    content = decisions_file.read_text(encoding="utf-8")

    required_decisions = [
        "Decision 007",
        "Decision 008",
        "Decision 009",
        "Decision 010",
        "Decision 011",
        "Decision 012",
        "Decision 013",
        "EXP_A_PRICE",
        "XGBoost",
        "Market Regime",
        "Wilcoxon",
        "CryptocurrencyPrediction",
    ]

    for dec in required_decisions:
        assert dec in content, f"Missing required decision item '{dec}' in DECISIONS.md"

    assert "scale-stable" in content.lower(), "DECISIONS.md must contain scale-stable target terminology."
    assert "H = 12" in content or "H=12" in content, "DECISIONS.md must document horizon H=12 in Decision 013."
