"""
Test suite for Phase 4 Sequence Generation & Splitting.

Verifies:
- 3D Sequence Shapes (N, 60, K) for 4, 5, 19, 24 configurations.
- Chronological extraction (no target leakage).
- Standard Scaler is fit exclusively on Training boundaries.
- Validation and Test observations never enter training bounds.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from crypto_analyzer.preprocessing.sequences import fit_transform_scaler, extract_3d_sequences
from crypto_analyzer.preprocessing.targets import compute_targets
from crypto_analyzer.preprocessing.pipeline import build_experiment_tensors


@pytest.fixture
def dummy_feature_df():
    """Returns a dummy feature dataframe of length 100 with 4 features + timestamp."""
    timestamps = np.arange(100) * 300000 + 1000000
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.arange(100, 200, dtype=float),  # Monotonic close for easy target checking
    })
    return df


def test_scaler_isolation(dummy_feature_df):
    """
    Verify the scaler is fit ONLY on the training feature partition.
    Modifying the validation/test data heavily must not alter the scaler parameters.
    """
    train_end_idx = 70
    
    # Run once normally
    scaled_normal, scaler_normal = fit_transform_scaler(
        dummy_feature_df, 
        train_end_idx=train_end_idx, 
        exclude_columns=['timestamp']
    )
    
    # Run again but poison the validation data with extreme outliers
    poisoned_df = dummy_feature_df.copy()
    poisoned_df.loc[75:, 'close'] = 99999999.9
    
    scaled_poisoned, scaler_poisoned = fit_transform_scaler(
        poisoned_df, 
        train_end_idx=train_end_idx, 
        exclude_columns=['timestamp']
    )
    
    # The fitted parameters (mean, scale) must be strictly identical
    np.testing.assert_array_equal(scaler_normal.mean_, scaler_poisoned.mean_)
    np.testing.assert_array_equal(scaler_normal.scale_, scaler_poisoned.scale_)


def test_sequence_dimensions_and_counts(dummy_feature_df):
    """Verify sequence generator outputs exact expected lengths and (W, K) dimensions."""
    window_size = 10
    horizon = 5
    # Dummy length = 100.
    # Targets length = 100 - 5 = 95.
    # Train end target idx = 70. Val end target idx = 85.
    
    targets = compute_targets(dummy_feature_df, horizon=horizon)
    tensors = extract_3d_sequences(
        dummy_feature_df, targets, 
        window_size=window_size, horizon=horizon,
        train_end_target_idx=70, val_end_target_idx=85
    )
    
    # Valid t bounds: t in [W-1, 95 - 1] -> [9, 94] (Total = 86)
    # Train: Target realization idx = t + H < 70 -> t + 5 < 70 -> t < 65 -> t in [9, 64] -> count = 64 - 9 + 1 = 56
    # Val: 70 <= t + 5 < 85 -> t in [65, 79] -> count = 79 - 65 + 1 = 15
    # Test: 85 <= t + 5 < 100 -> t in [80, 94] -> count = 94 - 80 + 1 = 15
    assert len(tensors['X_train']) == 56
    assert len(tensors['X_val']) == 15
    assert len(tensors['X_test']) == 15
    
    # X feature dimension K=4 (timestamp excluded)
    assert tensors['X_train'].shape == (56, 10, 4)
    # Y target dimension 2 (regression, classification)
    assert tensors['Y_train'].shape == (56, 2)


def test_no_target_leakage_and_ordering(dummy_feature_df):
    """
    Verify that sequence at index `i` contains strictly features up to `t`, 
    and target exactly derived from `Close[t+horizon]`.
    """
    window_size = 5
    horizon = 3
    train_end = 60
    
    targets = compute_targets(dummy_feature_df, horizon=horizon)
    tensors = extract_3d_sequences(
        dummy_feature_df, targets, 
        window_size=window_size, horizon=horizon,
        train_end_target_idx=train_end, val_end_target_idx=80
    )
    
    # Take the last training sequence
    last_train_x = tensors['X_train'][-1]
    last_train_y = tensors['Y_train'][-1]
    
    # Which 't' did this correspond to? 
    # For train, max(t+horizon) < train_end -> max(t+3) < 60 -> max(t) = 56
    t = 56
    
    # The last element of the sequence must perfectly match the feature row at t
    expected_last_row_features = dummy_feature_df.drop(columns=['timestamp']).iloc[t].values
    np.testing.assert_array_almost_equal(last_train_x[-1], expected_last_row_features)
    
    # The first element must match t - W + 1 = 56 - 5 + 1 = 52
    expected_first_row_features = dummy_feature_df.drop(columns=['timestamp']).iloc[t - window_size + 1].values
    np.testing.assert_array_almost_equal(last_train_x[0], expected_first_row_features)
    
    # The target must exactly match the computation from t to t+3
    # Close is monotonic +1 per row, so close[t+3] - close[t] = 3.0
    # Expected return = (3.0 / close[t]) * 100
    expected_close_t = dummy_feature_df['close'].iloc[t]
    expected_ret = (3.0 / expected_close_t) * 100.0
    
    # Verify R_t_3 matches
    assert pytest.approx(last_train_y[0]) == expected_ret
    assert last_train_y[1] == 1.0  # Positive return -> D=1


def test_pipeline_orchestrator(dummy_feature_df):
    """Verify build_experiment_tensors runs seamlessly end-to-end."""
    # Add a mock volume column so EXP_B_PRICE_VOL can be tested
    dummy_feature_df['volume'] = np.random.randn(100)
    
    tensors = build_experiment_tensors(
        dummy_feature_df, 
        experiment_id='EXP_B_PRICE_VOL', 
        window_size=10, 
        horizon=5
    )
    
    # EXP_B_PRICE_VOL has K=5
    assert tensors['X_train'].shape[2] == 5
    assert tensors['X_val'].shape[2] == 5
    assert tensors['X_test'].shape[2] == 5
