"""
Description :   Implements various imputation techniques (Mean, Median, Mode, KNN).
                Lets you apply different methods column-wise.
"""
import pandas as pd
from sklearn.impute import KNNImputer
import random

def fill_with_random_values(df: pd.DataFrame, col=None) -> pd.DataFrame:
    """
    Fills missing values in all columns
    with random values from that column's existing unique values.
    """

    unique_vals = df[col].dropna().unique().tolist()
    df[col] = df[col].apply(
        lambda x: random.choice(unique_vals) if pd.isnull(x) else x
    )

    return df


def fill_numeric_with_knn_imputer(df: pd.DataFrame, col=None) -> pd.DataFrame:
    """
    Fills missing values in numeric columns using KNN imputation.
    Automatically chooses n_neighbors based on dataset size.

    Parameters:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: DataFrame with numeric columns imputed.
    """

    # Auto-select n_neighbors: 5% of rows, at least 2, at most 5
    n_neighbors = min(5, max(2, int(len(df) * 0.05)))

    # Apply KNN only on numeric columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_values = imputer.fit_transform(df[[col]])
    df[col] = imputed_values

    return df


def fill_numeric_columns(df: pd.DataFrame, col=None, method: str = 'mean') -> pd.DataFrame:
    """
    Fills missing values in numeric (int/float) columns using the specified method.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - method (str): Method to fill missing values. Options: 'mean', 'median', or 'mode'.

    Returns:
    - pd.DataFrame: DataFrame with missing values in numeric columns filled.
    """


    if df[col].isnull().any():
        if method == 'mean':
            value = df[col].mean()
        elif method == 'median':
            value = df[col].median()
        elif method == 'mode':
            value = df[col].mode()[0]
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'mean', 'median', or 'mode'.")
        df[col].fillna(value, inplace=True)

    return df

def fill_missing_object_columns(df: pd.DataFrame, col=None, method = 'mode') -> pd.DataFrame:
    """
    Fills missing values in object-type columns using the specified method.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - method (str): Method to fill missing values. Options:
        - 'mode': Replaces missing values with the column's mode.
        - 'new_category': Replaces missing values with the string 'Missing'.

    Returns:
    - pd.DataFrame: A new dataframe with missing values filled in object columns.
    """

    if df[col].isnull().any():
        if method == 'mode':
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        elif method == 'new_category':
            df[col].fillna('Missing', inplace=True)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'mode' or 'new_category'.")
    return df
