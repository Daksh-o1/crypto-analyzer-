#!/usr/bin/env python
"""
Phase 8: Market Regime Analysis CLI script.

Reads the processed feature CSV, computes rolling market regimes,
and writes result artifacts to experiments/results/ and reports/.

Usage:
    python scripts/run_regime_analysis.py

All parameters are derived from the documented methodology in RESEARCH.md §8
and DECISIONS.md §017.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path when running as script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto_analyzer.regimes.analysis import run_regime_analysis

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "btcusdt_5m_features.csv"
OUTPUT_DIR_RESULTS = ROOT / "experiments" / "results"
OUTPUT_DIR_REPORTS = ROOT / "reports" / "tables"


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Market Regime Analysis")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help="Path to processed feature CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR_RESULTS,
        help="Output directory for JSON artifacts",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=OUTPUT_DIR_REPORTS,
        help="Output directory for markdown report",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Processed data file not found: {args.data}")
        sys.exit(1)

    print(f"[Phase 8] Loading processed data from: {args.data}")
    print(f"[Phase 8] Using regime window W_regime = 288 candles (24 hours)")
    print(f"[Phase 8] Theta slope = 0.0005 (normalized per 5-min step)")
    print(f"[Phase 8] Theta vol_high = 90th percentile of training-partition volatility")

    summary = run_regime_analysis(
        processed_data_path=args.data,
        output_dir=args.output,
    )

    # Also write the markdown report to reports/tables/
    from crypto_analyzer.regimes.analysis import _write_regime_report
    args.reports.mkdir(parents=True, exist_ok=True)
    _write_regime_report(summary, args.reports / "regime_analysis_report.md")

    test_analysis = summary["test_partition_analysis"]
    print("\n[Phase 8] === TEST PARTITION REGIME COUNTS ===")
    for regime, count in test_analysis["regime_counts"].items():
        prop = test_analysis["regime_proportions"].get(regime, 0.0)
        print(f"  {regime:<18}: {count:>5} ({prop:.1%})")

    print(f"\n[Phase 8] Artifacts written to:")
    print(f"  {args.output / 'regime_analysis_summary.json'}")
    print(f"  {args.output / 'test_regime_labels.csv'}")
    print(f"  {args.reports / 'regime_analysis_report.md'}")
    print("[Phase 8] Complete.")


if __name__ == "__main__":
    main()
