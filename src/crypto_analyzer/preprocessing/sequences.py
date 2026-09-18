"""
Sequence construction module for Phase 4 Target & Sequence Pipeline.

Handles the creation of sliding-window 3D tensors from 2D features,
strict temporal partitioning (Train, Validation, Test), and proper
scaler fitting (Train-only parameters).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_transform_scaler(
    df: pd.DataFrame, 
    train_end_idx: int,
    exclude_columns: List[str] = ['timestamp']
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fits a StandardScaler purely on the training feature partition, 
    and transforms the entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features.
    train_end_idx : int
        The row index strictly defining the end of the training partition.
    exclude_columns : List[str]
        Columns to ignore when scaling (e.g., timestamps).

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled DataFrame (as a copy) and the fitted scaler object.
    """
    scaled_df = df.copy()
    feature_cols = [c for c in df.columns if c not in exclude_columns]

    scaler = StandardScaler()
    
    # Fit strictly on the training slice
    train_features = df.iloc[:train_end_idx][feature_cols]
    scaler.fit(train_features)

    # Transform the whole dataset
    scaled_df[feature_cols] = scaler.transform(df[feature_cols])

    return scaled_df, scaler


def extract_3d_sequences(
    features_df: pd.DataFrame, 
    targets_df: pd.DataFrame,
    window_size: int = 60,
    horizon: int = 12,
    train_end_target_idx: int = 5980,
    val_end_target_idx: int = 7262
) -> Dict[str, np.ndarray]:
    """
    Creates chronologically ordered 3D sequences and partitions them based 
    on the underlying feature bounds of their respective targets, preventing
    cross-split target leakage.

    Parameters
    ----------
    features_df : pd.DataFrame
        Fully populated (post-warm-up) and scaled features (N rows).
    targets_df : pd.DataFrame
        Corresponding targets (N - H rows). Row i contains target for i+H.
    window_size : int
        Lookback window size (W).
    horizon : int
        Prediction horizon (H).
    train_end_target_idx : int
        Boundary index in features_df denoting end of training features.
    val_end_target_idx : int
        Boundary index in features_df denoting end of validation features.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing X and Y partitions:
        'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test'
    """
    if len(features_df) - horizon != len(targets_df):
        raise ValueError(f"Features and Targets length mismatch. Expected features {len(features_df)} - {horizon} == {len(targets_df)}")

    # Exclude timestamp from features matrix but keep it aligned if needed
    feature_cols = [c for c in features_df.columns if c != 'timestamp']
    feat_matrix = features_df[feature_cols].values
    
    target_cols = [c for c in targets_df.columns if c != 'timestamp']
    targ_matrix = targets_df[target_cols].values

    # Determine dimensions
    N_valid = len(targets_df)
    K = feat_matrix.shape[1]

    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    # Iterate through every valid sequence-end index
    for t in range(window_size - 1, N_valid):
        # The sequence requires W observations ending at t
        x_seq = feat_matrix[t - window_size + 1 : t + 1]
        
        # The target corresponding to the sequence ending at t
        # targets_df aligns row-by-row with features_df, so index t is target evaluated at t+H
        y_seq = targ_matrix[t]
        
        # Determine Partition Assignment based on future target realization boundary
        target_realization_idx = t + horizon
        
        if target_realization_idx < train_end_target_idx:
            train_X.append(x_seq)
            train_Y.append(y_seq)
        elif target_realization_idx < val_end_target_idx:
            val_X.append(x_seq)
            val_Y.append(y_seq)
        else:
            test_X.append(x_seq)
            test_Y.append(y_seq)

    return {
        'X_train': np.array(train_X),
        'Y_train': np.array(train_Y),
        'X_val': np.array(val_X),
        'Y_val': np.array(val_Y),
        'X_test': np.array(test_X),
        'Y_test': np.array(test_Y)
    }
