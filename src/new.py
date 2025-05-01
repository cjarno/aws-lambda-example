import pandas as pd
import numpy as np
from loguru import logger

from src.AlphaVantage import load_top10_annualised_data
from src.MonteCarlo import MonteCarloSimulation, MonteCarloSettings
from src.utility import log_optimal_portfolio


annualised_top10_data = load_top10_annualised_data()

mcs_settings = MonteCarloSettings(
    iterations=1_000, dimensions=10, weight_normalisation=True
)


def portfolio_optimisation_metrics_method(
    random_vector: np.ndarray,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    mean_rfr: float,
) -> np.ndarray:
    """Calculates portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = np.dot(random_vector, mean_returns)
    portfolio_volatility = np.sqrt(random_vector.T @ covariance_matrix @ random_vector)
    portfolio_sharpe = (portfolio_return - mean_rfr) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, portfolio_sharpe])


mcs = MonteCarloSimulation(settings=mcs_settings)
logger.info(mcs.settings)

metric_keys = ("returns", "volatility", "sharpe")
optimal_key = "sharpe"

mcs.set_simulation_function(
    simulation_func=portfolio_optimisation_metrics_method, metric_count=len(metric_keys)
)
mcs.set_simulation_inputs(
    element_labels=tuple(annualised_top10_data["instruments"]),
    mean_returns=annualised_top10_data["mean_returns"],
    covariance_matrix=annualised_top10_data["covariance_matrix"],
    mean_rfr=annualised_top10_data["mean_rfr"],
)
mcs.run()

optimal_vector = mcs.get_optimal_vector(
    metric_keys=metric_keys, max_by_metric=optimal_key
)

log_optimal_portfolio(optimal_vector)

logger.success("----DONE----")
