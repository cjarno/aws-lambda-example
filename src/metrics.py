import pandas as pd
import numpy as np


def calculate_portfolio_metrics(
    random_vector: np.ndarray,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    mean_rfr: float,
) -> np.ndarray:
    """Calculates expected return, volatility, and Sharpe ratio for a
    portfolio based on a given weight vector, mean returns, covariance matrix, and
    risk-free rate.

    Args:
        random_vector (np.ndarray): Array of portfolio weights.
        mean_returns (pd.Series): Expected returns for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns, shape
            (n_assets, n_assets).
        mean_rfr (float): Mean risk-free rate for Sharpe ratio calculation.

    Returns:
        np.ndarray: Array of [return, volatility, sharpe]

    Raises:
        ValueError: If the shapes of `random_vector`, `mean_returns`, or
            `covariance_matrix` are incompatible.
    """
    n_assets = len(mean_returns)
    if len(random_vector) != n_assets:
        raise ValueError("random_vector and mean_returns have different lengths.")

    if covariance_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            "covariance_matrix is not square or mismatches mean_returns length."
        )

    portfolio_return = np.dot(random_vector, mean_returns)
    portfolio_volatility = np.sqrt(random_vector.T @ covariance_matrix @ random_vector)
    portfolio_sharpe = (portfolio_return - mean_rfr) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, portfolio_sharpe])
