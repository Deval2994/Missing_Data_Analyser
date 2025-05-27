"""
Description :   Load data and find insight like:
            *    1. Total Missing Values (by column and overall)
            *    2. Percentage of missing data
            *    3. Heatmap/Matrix data for visualization
            *    4. How many rows would be lost with CCA
            *    5. import MCAR from mcar_test.py
            *    6. filling missing data
            *    7. data cleaning by CCA
"""

import pandas as pd
import numpy as np

def nan_decoding(df: pd.DataFrame, nan_indicator=None):
    # Step 1: Replace common missing indicators with NaN
    if nan_indicator is None:
        nan_indicator = ["?", "NA", "N/A", "na", "--", "null", "None", ""]

    # Step 2: Attempt to convert each column to numeric or datetime
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    df = df.replace(nan_indicator, np.nan)

    return df

