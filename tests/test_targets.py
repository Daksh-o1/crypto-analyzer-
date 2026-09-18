"""
Test suite for Phase 4 Target Generation.

Verifies:
- Exact regression formula R(t, 12).
- Exact classification logic D(t, 12) (including boundary 0).
- Truncation of final H rows to prevent future fabrication.
- Timestamp alignment preservation.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.preprocessing.targets import compute_targets


@pytest.fixture
def mock_feature_df():
    """Returns a mock feature dataframe with 20 rows."""
    timestamps = list(range(1000000, 1000000 + 20 * 300000, 300000))
    # Close prices structured to test positive, negative, and zero returns
    close = [
        100.0, 105.0, 95.0, 100.0, 110.0, 
        100.0, 120.0, 100.0, 100.0, 80.0,
        100.0, 100.0, 110.0, 115.0, 85.0,
        100.0, 105.0, 120.0, 100.0, 100.0
    ]
    return pd.DataFrame({
        'timestamp': timestamps,
        'close': close,
        'other_feature': np.random.randn(20)
    })


def test_regression_target_formula(mock_feature_df):
    """Verify R(t, 12) = ((Close[t+12] - Close[t]) / Close[t]) * 100."""
    # Using a small horizon = 3 for this 20-row dataframe to easily verify
    horizon = 3
    targets = compute_targets(mock_feature_df, horizon=horizon)
    
    # At t=0, Close[0] = 100.0. Close[0+3] = Close[3] = 100.0. Return = 0.0
    assert pytest.approx(targets['R_t_3'].iloc[0]) == 0.0
    
    # At t=1, Close[1] = 105.0. Close[1+3] = Close[4] = 110.0. Return = 5/105 * 100
    expected_ret_1 = ((110.0 - 105.0) / 105.0) * 100
    assert pytest.approx(targets['R_t_3'].iloc[1]) == expected_ret_1
    
    # At t=2, Close[2] = 95.0. Close[2+3] = Close[5] = 100.0. Return = 5/95 * 100
    expected_ret_2 = ((100.0 - 95.0) / 95.0) * 100
    assert pytest.approx(targets['R_t_3'].iloc[2]) == expected_ret_2


def test_classification_target_logic(mock_feature_df):
    """Verify D(t, 12) = 1 if R > 0 else 0."""
    horizon = 3
    targets = compute_targets(mock_feature_df, horizon=horizon)
    
    # t=0, ret=0.0 -> class=0
    assert targets['D_t_3'].iloc[0] == 0
    # t=1, ret>0 -> class=1
    assert targets['D_t_3'].iloc[1] == 1
    # t=11, Close[11]=100, Close[14]=85 -> ret<0 -> class=0
    assert targets['D_t_3'].iloc[11] == 0


def test_final_h_rows_dropped(mock_feature_df):
    """Verify the final H rows are explicitly removed."""
    horizon = 5
    original_len = len(mock_feature_df)
    targets = compute_targets(mock_feature_df, horizon=horizon)
    
    assert len(targets) == original_len - horizon


def test_target_timestamp_alignment(mock_feature_df):
    """Verify that timestamp t in targets maps to timestamp t in features."""
    horizon = 4
    targets = compute_targets(mock_feature_df, horizon=horizon)
    
    # The timestamps in the resulting target df should perfectly match the first (N-H) timestamps of input
    expected_timestamps = mock_feature_df['timestamp'].iloc[:-horizon].values
    np.testing.assert_array_equal(targets['timestamp'].values, expected_timestamps)


def test_no_fabricated_targets(mock_feature_df):
    """Ensure we don't have NaN returns in the final output (no ffill/bfill fabrication)."""
    horizon = 2
    targets = compute_targets(mock_feature_df, horizon=horizon)
    
    assert targets['R_t_2'].isna().sum() == 0
    assert targets['D_t_2'].isna().sum() == 0


def test_insufficient_data_raises_error(mock_feature_df):
    """Ensure error is raised if dataframe length is <= horizon."""
    with pytest.raises(ValueError, match="must be greater than horizon"):
        compute_targets(mock_feature_df, horizon=20)


def test_missing_required_columns(mock_feature_df):
    """Ensure error is raised if 'close' or 'timestamp' is missing."""
    bad_df = mock_feature_df.drop(columns=['close'])
    with pytest.raises(ValueError, match="contain 'close' and 'timestamp'"):
        compute_targets(bad_df, horizon=2)
