"""
Description :   Performs Little’s MCAR test to statistically test whether missing values are
                Missing Completely At Random (MCAR), helping justify if CCA is a valid choice.
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance


def is_mcar(data: pd.DataFrame) -> bool:
    """
    Approximate Little's MCAR test.

    Parameters:
    - data: pd.DataFrame with missing values

    Returns:
    - True if data is likely MCAR (p-value > 0.05)
    - False if data is likely NOT MCAR (p-value <= 0.05)

    Raises:
    - ValueError if not enough missing data patterns or data to perform test.
    """

    # Step 1: Keep only columns with at least some missing data but not all missing
    df = data.copy()
    df = df.loc[:, df.isnull().any() & ~df.isnull().all()]

    if df.shape[1] < 2:
        raise ValueError("Need at least two columns with missing values for MCAR test.")

    # Step 2: Create a matrix indicating missingness (1 = missing, 0 = observed)
    missing_indicator = df.isnull().astype(int)

    # Step 3: Identify unique missing data patterns in the dataset
    unique_patterns = missing_indicator.drop_duplicates()

    test_statistic = 0
    degrees_of_freedom = 0

    # Step 4: For each missing data pattern group:
    for _, pattern_row in unique_patterns.iterrows():
        # Find rows matching this missing pattern
        mask = (missing_indicator == pattern_row.values).all(axis=1)
        group = df[mask]

        # Skip groups with too few rows to analyze
        if len(group) < 2:
            continue

        # Step 5: Select only observed (non-missing) columns for this group
        observed_data = group.dropna(axis=1)

        # Need at least two columns observed to calculate covariance
        if observed_data.shape[1] < 2:
            continue

        try:
            # Calculate mean vector of observed data
            mean_vector = observed_data.mean().values

            # Calculate covariance matrix of observed data
            cov_matrix = EmpiricalCovariance().fit(observed_data).covariance_

            # Calculate inverse of covariance matrix (use pseudo-inverse for stability)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)

            # Center the observed data by subtracting mean vector
            centered_data = observed_data - mean_vector

            # Compute Mahalanobis distances squared for all rows in the group
            # Using einsum for efficient matrix multiplication
            mahalanobis_sq = np.einsum('ij,jk,ik->i', centered_data.values, inv_cov_matrix, centered_data.values)

            # Sum of Mahalanobis distances squared contributes to test statistic
            test_statistic += mahalanobis_sq.sum()

            # Degrees of freedom accumulate: number of observations * number of variables
            degrees_of_freedom += observed_data.shape[0] * observed_data.shape[1]

        except Exception as e:
            # If any error occurs (e.g., singular matrix), skip this group
            continue

    # Step 6: Check if we have valid degrees of freedom to perform test
    if degrees_of_freedom == 0:
        raise ValueError("Insufficient complete data to perform MCAR test.")

    # Step 7: Calculate p-value from Chi-squared distribution
    p_value = 1 - chi2.cdf(test_statistic, df=degrees_of_freedom)

    # Step 8: Return True if p-value > 0.05 (fail to reject MCAR), else False
    return p_value > 0.05
