from loguru import logger

from src.AlphaVantage import load_top10_annualised_data
from src.MonteCarlo import MonteCarloSimulation, MonteCarloSettings
from src.metrics import calculate_portfolio_metrics
from src.utility import log_optimal_portfolio


if __name__ == '__main__':
    annualised_top10_data = load_top10_annualised_data()

    mcs_settings = MonteCarloSettings(
        iterations=10_000, dimensions=10, weight_normalisation=True
    )

    mcs = MonteCarloSimulation(settings=mcs_settings)
    logger.info(mcs.settings)

    metric_keys = ("returns", "volatility", "sharpe")
    optimal_key = "sharpe"

    mcs.set_simulation_function(
        simulation_func=calculate_portfolio_metrics, metric_count=len(metric_keys)
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
