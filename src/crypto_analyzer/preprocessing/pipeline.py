"""
Phase 4 Orchestrator Pipeline.

Coordinates the loading of feature matrices, removal of warm-up rows,
calculation of strict time-series targets, chronological partitioning,
train-only scaler transformations, and 3D sequence building for all 
experiment layouts (EXP_A, B, C, D).
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from crypto_analyzer.features.pipeline import get_feature_columns
from crypto_analyzer.preprocessing.targets import compute_targets
from crypto_analyzer.preprocessing.sequences import fit_transform_scaler, extract_3d_sequences


def build_experiment_tensors(
    df: pd.DataFrame,
    experiment_id: str,
    window_size: int = 60,
    horizon: int = 12
) -> Dict[str, np.ndarray]:
    """
    Build 3D sequence tensors for a specific experiment layout.

    Parameters
    ----------
    df : pd.DataFrame
        Processed Phase 3 dataframe containing all 24 features and timestamp.
    experiment_id : str
        The experiment layout identifier (e.g., 'EXP_D_FULL').
    window_size : int
        Lookback window size (W).
    horizon : int
        Prediction horizon (H).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with scaled 3D tensors:
        'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test'
    """
    # 1. Filter columns to the specified experiment layout
    expected_cols = ['timestamp'] + get_feature_columns(experiment_id)
    # Note: get_feature_columns might not return them with 'timestamp', but the df has it.
    
    # Ensure no duplicates in column selection
    expected_cols = list(dict.fromkeys(expected_cols))
    
    # Extract dataset subset for this experiment
    exp_df = df[expected_cols].copy()

    # 2. Drop Phase 3 warm-up rows globally
    # The first 96 rows typically contain NaNs from moving averages
    exp_df = exp_df.dropna().reset_index(drop=True)

    # 3. Compute partition boundaries purely on the available feature rows
    total_valid_rows = len(exp_df)
    train_end_idx = int(total_valid_rows * 0.70)
    val_end_idx = int(total_valid_rows * 0.85)

    # 4. Generate Strict Targets
    # Note: this drops the last 12 rows internally, returning len(exp_df) - 12
    targets_df = compute_targets(exp_df, horizon=horizon)

    # 5. Fit & Apply Scaler (strictly fitting on Train split only)
    # The target columns are entirely isolated in targets_df and not part of scaling
    scaled_features_df, _ = fit_transform_scaler(
        df=exp_df,
        train_end_idx=train_end_idx,
        exclude_columns=['timestamp']
    )

    # 6. Build Sequences & Distribute chronologically without target leakage
    tensors = extract_3d_sequences(
        features_df=scaled_features_df,
        targets_df=targets_df,
        window_size=window_size,
        horizon=horizon,
        train_end_target_idx=train_end_idx,
        val_end_target_idx=val_end_idx
    )

    return tensors
