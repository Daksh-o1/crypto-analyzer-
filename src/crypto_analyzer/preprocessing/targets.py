"""
Target generation module for Phase 4 Sequence Pipeline.

Computes mathematically strict prediction targets matching the 
1-hour horizon (H=12) specification. Zero future data leaks into 
the feature set itself; target values are strictly maintained in 
separate target arrays aligned to the original timestamp t.
"""

import numpy as np
import pandas as pd


def compute_targets(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """
    Generate regression and classification targets for a given horizon.

    Calculates the relative percentage return from timestamp t to t+H,
    and a binary classification for positive directional movement.

    The final `horizon` rows of the DataFrame will not have a valid
    target and are explicitly removed to prevent fabricating future data.

    Parameters
    ----------
    df : pd.DataFrame
        Processed features dataframe. Must contain 'timestamp' and 'close'.
    horizon : int
        Number of steps ahead for the prediction horizon (default 12 = 1 hour).

    Returns
    -------
    pd.DataFrame
        Target dataframe with columns ['timestamp', 'R_t_{horizon}', 'D_t_{horizon}'].
        Length will be exactly len(df) - horizon.

    Raises
    ------
    ValueError
        If required columns are missing, or the dataframe is too short.
    """
    if 'close' not in df.columns or 'timestamp' not in df.columns:
        raise ValueError("Input dataframe must contain 'close' and 'timestamp' columns.")

    if len(df) <= horizon:
        raise ValueError(f"Dataframe length ({len(df)}) must be greater than horizon ({horizon}).")

    close = df['close']
    
    # Calculate R(t, H) = ((Close[t+H] - Close[t]) / Close[t]) * 100
    future_close = close.shift(-horizon)
    returns = ((future_close - close) / close) * 100.0

    # Calculate D(t, H) = 1 if R(t, H) > 0 else 0
    # Must explicitly handle NaN cases to avoid evaluating to False (0) for missing targets
    direction = (returns > 0.0).astype(float)
    
    # Assign NaNs to the directional target where returns are NaN
    direction[returns.isna()] = np.nan

    target_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        f'R_t_{horizon}': returns,
        f'D_t_{horizon}': direction
    }, index=df.index)

    # Explicitly drop the final `horizon` rows which have no valid future value
    target_df = target_df.dropna(subset=[f'R_t_{horizon}'])

    # Cast classification target back to integer for cleanliness
    target_df[f'D_t_{horizon}'] = target_df[f'D_t_{horizon}'].astype(int)

    return target_df
