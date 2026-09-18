"""
Phase 8 regime analysis orchestrator.

Loads the processed feature data, computes rolling market regimes on the
full dataset (aligned to the raw feature rows), and then reports regime
statistics specifically on the TEST partition — the only partition used
for out-of-sample model evaluation.

Leakage safeguard:
  theta_vol_high is derived strictly from training-partition volatility
  values (rows 0..train_end_idx), mirroring the train-only scaler policy
  from Phase 4 / DECISIONS.md §017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from crypto_analyzer.regimes.classifier import (
    REGIME_WINDOW,
    THETA_SLOPE,
    WARMUP_LABEL,
    compute_rolling_regimes,
)


# ---------------------------------------------------------------------------
# Constants (mirrors Phase 4 partitioning from DECISIONS.md §017)
# ---------------------------------------------------------------------------

TOTAL_FEATURE_ROWS: int = 8544   # After warm-up removal (8640 raw - 96 warm-up)
TRAIN_END_IDX: int = int(TOTAL_FEATURE_ROWS * 0.70)   # = 5980
VAL_END_IDX: int = int(TOTAL_FEATURE_ROWS * 0.85)     # = 7262
# Test partition: rows 7262..8543 (1282 rows)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_regime_analysis(
    processed_data_path: Path,
    output_dir: Path,
    window: int = REGIME_WINDOW,
    theta_slope: float = THETA_SLOPE,
    vol_percentile: float = 90.0,
) -> Dict:
    """
    Run the Phase 8 market regime analysis pipeline.

    Parameters
    ----------
    processed_data_path : Path
        Path to the processed feature CSV (all 24 features, post-warm-up).
    output_dir : Path
        Directory to write regime result artifacts.
    window : int
        Regime rolling window (288 candles = 24 h).
    theta_slope : float
        Normalized slope threshold.
    vol_percentile : float
        Percentile for computing theta_vol_high from training volatility.

    Returns
    -------
    Dict
        Summary statistics and artifact paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load processed feature data
    df = pd.read_csv(processed_data_path)
    # Ensure chronological order
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Drop the warm-up NaN rows (mirrors Phase 4 pipeline)
    df = df.dropna().reset_index(drop=True)
    n_feature_rows = len(df)

    # 3. Compute rolling regimes on the full feature-row Close series
    close_series = df["close"].reset_index(drop=True)
    regime_df, theta_vol_high, theta_slope_used = compute_rolling_regimes(
        close=close_series,
        window=window,
        theta_slope=None,             # derive data-adaptively from training
        theta_vol_high=None,          # derive from training data
        vol_percentile=vol_percentile,
        slope_percentile=50.0,        # median of training abs(slope)
        train_end_idx=TRAIN_END_IDX,
    )

    # Attach regime info back to the feature rows
    df["regime"] = regime_df["regime"].values
    df["regime_slope"] = regime_df["slope"].values
    df["regime_volatility"] = regime_df["volatility"].values
    df["regime_code"] = regime_df["regime_code"].values

    # 4. Partition indices
    train_end = TRAIN_END_IDX
    val_end = VAL_END_IDX

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    n_test = len(df_test)

    # 5. Test-partition regime analysis (primary analysis — out-of-sample)
    test_regimes = df_test["regime"]
    test_valid = df_test[df_test["regime"] != WARMUP_LABEL]
    n_test_classified = len(test_valid)

    regime_names = ["Bullish", "Bearish", "Sideways", "High Volatility"]
    test_counts = {r: int((test_regimes == r).sum()) for r in regime_names + [WARMUP_LABEL]}
    test_proportions = {r: round(c / n_test, 4) for r, c in test_counts.items()}

    # 6. Test-partition: basic market stats by regime
    regime_stats = {}
    for regime in regime_names:
        mask = df_test["regime"] == regime
        subset = df_test[mask]
        if len(subset) == 0:
            regime_stats[regime] = {"count": 0}
            continue
        # Simple returns within this regime subset
        if "close" in subset.columns:
            returns = subset["close"].pct_change().dropna()
            regime_stats[regime] = {
                "count": int(mask.sum()),
                "mean_close": round(float(subset["close"].mean()), 4),
                "std_close": round(float(subset["close"].std()), 4),
                "mean_return_pct": round(float(returns.mean() * 100), 6),
                "std_return_pct": round(float(returns.std() * 100), 6),
                "mean_slope": round(float(subset["regime_slope"].mean()), 8),
                "mean_volatility": round(float(subset["regime_volatility"].mean()), 8),
            }

    # 7. Full-dataset regime counts (including train/val for context)
    full_counts = {r: int((df["regime"] == r).sum()) for r in regime_names + [WARMUP_LABEL]}

    # 8. Chronological distribution — regime per 100-row block in test partition
    block_size = 100
    chrono_blocks = []
    for start in range(0, n_test, block_size):
        block = df_test.iloc[start : start + block_size]
        dominant = block["regime"].mode()
        dominant_regime = dominant.iloc[0] if len(dominant) > 0 else "Unknown"
        chrono_blocks.append(
            {
                "block_start": start,
                "block_end": min(start + block_size, n_test) - 1,
                "dominant_regime": dominant_regime,
                "regime_counts": {r: int((block["regime"] == r).sum()) for r in regime_names},
            }
        )

    # 9. Construct output summary
    summary = {
        "methodology": {
            "window_candles": window,
            "window_hours": window * 5 / 60,
            "theta_slope_documented": 0.0005,
            "theta_slope_used": round(theta_slope_used, 8),
            "theta_slope_derivation": "50th percentile of abs(slope) over training partition (rows 0..5979) — data-adaptive, train-only",
            "theta_vol_high": round(theta_vol_high, 8),
            "vol_percentile_source": "training partition only (rows 0..5979)",
            "vol_percentile_used": vol_percentile,
            "warm_up_rows_regime": window - 1,
            "regime_labels": ["Bullish", "Bearish", "Sideways", "High Volatility", "Warmup"],
        },
        "dataset_info": {
            "total_feature_rows": n_feature_rows,
            "train_rows": train_end,
            "val_rows": val_end - train_end,
            "test_rows": n_test,
        },
        "test_partition_analysis": {
            "n_test_total": n_test,
            "n_test_classified": n_test_classified,
            "n_test_warmup": test_counts.get(WARMUP_LABEL, 0),
            "regime_counts": test_counts,
            "regime_proportions": test_proportions,
            "regime_stats_by_label": regime_stats,
        },
        "full_dataset_regime_counts": full_counts,
        "chronological_blocks_test": chrono_blocks,
    }

    # 10. Save artifacts
    import json

    summary_path = output_dir / "regime_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save test-partition regime labels as CSV for downstream regime-conditioned analysis
    regime_csv_path = output_dir / "test_regime_labels.csv"
    df_test[["regime", "regime_slope", "regime_volatility", "regime_code"]].to_csv(
        regime_csv_path, index=True, encoding="utf-8"
    )

    # Generate markdown report
    _write_regime_report(summary, output_dir / "regime_analysis_report.md")

    return summary


def _write_regime_report(summary: Dict, path: Path) -> None:
    """Write a markdown descriptive report of the regime analysis."""
    md = summary["test_partition_analysis"]
    meta = summary["methodology"]
    ds = summary["dataset_info"]

    lines = [
        "# Phase 8: Market Regime Analysis Report",
        "",
        "## Methodology",
        f"- **Regime window (W_regime):** {meta['window_candles']} candles ({meta['window_hours']:.1f} hours)",
        f"- **Theta slope threshold (documented default):** {meta['theta_slope_documented']}",
        f"- **Theta slope threshold (used):** {meta['theta_slope_used']:.8f}",
        f"  *({meta['theta_slope_derivation']})*",
        f"- **Theta vol_high threshold:** {meta['theta_vol_high']:.8f}",
        f"  *(derived as {meta['vol_percentile_used']}th percentile of training-partition rolling volatility — {meta['vol_percentile_source']})*",
        f"- **Warm-up rows (regime):** {meta['warm_up_rows_regime']} rows (rows lacking a full {meta['window_candles']}-candle lookback are labelled 'Warmup')",
        "",
        "## Regime Classification Precedence",
        "1. **High Volatility** — sigma_t > theta_vol_high (regardless of slope)",
        "2. **Bullish** — S_t > +theta_slope AND sigma_t <= theta_vol_high",
        "3. **Bearish** — S_t < -theta_slope AND sigma_t <= theta_vol_high",
        "4. **Sideways** — |S_t| <= theta_slope AND sigma_t <= theta_vol_high",
        "",
        "## Dataset Partitions",
        f"| Partition | Rows |",
        f"|---|---|",
        f"| Train | {ds['train_rows']} |",
        f"| Validation | {ds['val_rows']} |",
        f"| Test | {ds['test_rows']} |",
        f"| Total Feature Rows | {ds['total_feature_rows']} |",
        "",
        "## Test Partition Regime Counts & Proportions",
        f"| Regime | Count | Proportion |",
        "|---|---|---|",
    ]

    for r in ["Bullish", "Bearish", "Sideways", "High Volatility", "Warmup"]:
        count = md["regime_counts"].get(r, 0)
        prop = md["regime_proportions"].get(r, 0.0)
        lines.append(f"| {r} | {count} | {prop:.2%} |")

    lines += [
        "",
        f"*(Out of {md['n_test_total']} test observations; {md['n_test_classified']} classified, {md['n_test_warmup']} in warm-up.)*",
        "",
        "## Descriptive Statistics by Regime (Test Partition)",
        "| Regime | Count | Mean Close | Mean Return % | Std Return % | Mean Slope | Mean Volatility |",
        "|---|---|---|---|---|---|---|",
    ]

    for r, stats in md["regime_stats_by_label"].items():
        if stats.get("count", 0) == 0:
            lines.append(f"| {r} | 0 | - | - | - | - | - |")
        else:
            lines.append(
                f"| {r} | {stats['count']} "
                f"| {stats.get('mean_close', 'N/A')} "
                f"| {stats.get('mean_return_pct', 'N/A'):.6f} "
                f"| {stats.get('std_return_pct', 'N/A'):.6f} "
                f"| {stats.get('mean_slope', 'N/A'):.8f} "
                f"| {stats.get('mean_volatility', 'N/A'):.8f} |"
            )

    lines += [
        "",
        "## Observations",
        "- Results are purely descriptive and observational.",
        "- No causal claims are made about regime types and model performance.",
        "- The theta_vol_high threshold was derived exclusively from training-partition data to prevent look-ahead leakage.",
        "- Regime labels are produced independently of any model predictions.",
        "- These regime labels are available for downstream use in Phase 9 (significance testing) and Phase 10 (visualization).",
        "",
        "## Limitations",
        "- Regime classification is based on a single rolling 24-hour window. Regime transitions may not be precisely aligned to exact structural breaks.",
        "- Short sideways or high-volatility episodes that are subsumed into adjacent longer regimes may be underrepresented.",
        "- The slope threshold theta_slope is a pre-specified constant rather than a data-adaptive threshold.",
        "- Warm-up rows at the start of the test partition (if any) are excluded from regime-conditioned analysis.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
