"""
Description :   Contains logic for Complete Case Analysis CCA
"""
import pandas as pd

def complete_case_analysis(data: pd.DataFrame):
    return data.dropna()

def data_loss_percentage(dataframe):
    """
    Returns the percentage of rows that would be dropped using Complete Case Analysis (CCA).

    Parameters:
    ----------
    dataframe : pd.DataFrame

    Returns:
    -------
    float : Data loss percentage due to CCA
    """
    rows_with_na = dataframe[dataframe.isnull().any(axis=1)]
    no_of_rows = rows_with_na.shape[0]
    total_rows = dataframe.shape[0]
    data_loss = (no_of_rows / total_rows) * 100
    return round(data_loss, 2)

def nan_percentage_per_column(dataframe):
    """
    Returns the percentage of NaN values for each column in the DataFrame.

    Parameters:
    ----------
    dataframe : pd.DataFrame

    Returns:
    -------
    pd.Series : Percentage of NaNs per column, rounded to 2 decimal places.
    """
    total_rows = len(dataframe)
    return round((dataframe.isnull().sum() / total_rows) * 100, 2)
