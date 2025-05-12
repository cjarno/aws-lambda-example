import pytest
import numpy as np
import pandas as pd
from src.MonteCarlo import MonteCarloSettings, MonteCarloSimulation


def mock_portfolio_function(
    weights: np.ndarray,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    mean_rfr: float,
) -> np.ndarray:
    """Mock function mimicking calculate_portfolio_metrics for testing.

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
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
    portfolio_sharpe = (portfolio_return - mean_rfr) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, portfolio_sharpe])


@pytest.fixture
def settings():
    """Fixture for MonteCarloSettings with default parameters."""
    return MonteCarloSettings(iterations=100, dimensions=5, weight_normalisation=True)


@pytest.fixture
def simulation(settings):
    """Fixture for MonteCarloSimulation instance."""
    return MonteCarloSimulation(settings)


@pytest.fixture
def mock_data():
    """Fixture for mock financial data."""
    dimensions = 5
    labels = [f"Asset{i}" for i in range(dimensions)]
    mean_returns = pd.Series(np.random.randn(dimensions), name="returns")
    covariance_matrix = pd.DataFrame(
        np.random.randn(dimensions, dimensions),
        columns=mean_returns.index,
        index=mean_returns.index,
    )
    covariance_matrix = (
        covariance_matrix @ covariance_matrix.T
    )  # Ensure positive semi-definite, given randomised values
    mean_rfr = 0.02
    return {
        "mean_returns": mean_returns,
        "covariance_matrix": covariance_matrix,
        "mean_rfr": mean_rfr,
        "element_labels": labels,
    }


@pytest.fixture
def configured_simulation(simulation, mock_data):
    """Fixture for a configured MonteCarloSimulation instance."""
    simulation.set_simulation_function(mock_portfolio_function, metric_count=3)
    simulation.set_simulation_inputs(**mock_data)
    return simulation


def test_monte_carlo_settings_init():
    """Test initialization of MonteCarloSettings."""
    settings = MonteCarloSettings(
        iterations=1000, dimensions=10, weight_normalisation=False
    )
    assert settings.iterations == 1000
    assert settings.dimensions == 10
    assert settings.weight_normalisation is False


def test_monte_carlo_settings_shape():
    """Test shape property of MonteCarloSettings."""
    settings = MonteCarloSettings(
        iterations=500, dimensions=3, weight_normalisation=True
    )
    assert settings.shape == (500, 3)


@pytest.mark.parametrize(
    "iterations,dimensions,weight_normalisation",
    [(0, 5, True), (-1, 5, False), (100, 0, True), (100, -1, False)],
)
def test_monte_carlo_settings_invalid_params(
    iterations, dimensions, weight_normalisation
):
    """Test MonteCarloSettings with invalid parameters."""
    with pytest.raises(ValueError):
        MonteCarloSettings(
            iterations=iterations,
            dimensions=dimensions,
            weight_normalisation=weight_normalisation,
        )


def test_monte_carlo_simulation_init(settings):
    """Test initialization of MonteCarloSimulation."""
    mcs = MonteCarloSimulation(settings)
    assert mcs.settings is settings
    assert mcs.simulation_func is None
    assert mcs.metric_count is None
    assert mcs.random_trials is None
    assert mcs.simulation_inputs is None
    assert mcs.results is None


def test_set_simulation_function(simulation):
    """Test set_simulation_function configures function and metric count."""
    simulation.set_simulation_function(mock_portfolio_function, metric_count=3)
    assert simulation.simulation_func == mock_portfolio_function
    assert simulation.metric_count == 3


@pytest.mark.parametrize("metric_count", [0, -1])
def test_set_simulation_function_invalid_metric_count(simulation, metric_count):
    """Test set_simulation_function with invalid metric_count."""
    with pytest.raises(ValueError, match="metric_count .* must be greater than 0"):
        simulation.set_simulation_function(
            mock_portfolio_function, metric_count=metric_count
        )


def test_generate_random_matrix(simulation):
    """Test _generate_random_matrix produces correct shape and normalization."""
    rng_matrix = simulation._generate_random_matrix()
    assert rng_matrix.shape == simulation.settings.shape
    assert np.all(rng_matrix >= 0) and np.all(rng_matrix <= 1)
    if simulation.settings.weight_normalisation:
        assert np.allclose(rng_matrix.sum(axis=1), 1.0)


def test_generate_random_matrix_no_normalization():
    """Test _generate_random_matrix without weight normalization."""
    settings = MonteCarloSettings(
        iterations=100, dimensions=5, weight_normalisation=False
    )
    simulation = MonteCarloSimulation(settings)
    rng_matrix = simulation._generate_random_matrix()
    assert rng_matrix.shape == (100, 5)
    assert not np.allclose(rng_matrix.sum(axis=1), 1.0)


def test_run_success(configured_simulation):
    """Test run method executes simulation and produces correct output."""
    results = configured_simulation.run()
    assert results.shape == (
        configured_simulation.settings.iterations,
        configured_simulation.metric_count,
    )
    assert configured_simulation.results.shape == results.shape
    assert (
        configured_simulation.random_trials.shape
        == configured_simulation.settings.shape
    )
    assert np.all(np.isfinite(results))


def test_run_missing_simulation_function(simulation, mock_data):
    """Test run raises error when simulation_func is not set."""
    simulation.set_simulation_inputs(**mock_data)
    with pytest.raises(ValueError):
        simulation.run()


def test_run_missing_simulation_inputs(simulation):
    """Test run raises error when simulation_inputs is not set."""
    simulation.set_simulation_function(mock_portfolio_function, metric_count=3)
    with pytest.raises(ValueError):
        simulation.run()


def test_get_optimal_vector_invalid_max_by_metric(configured_simulation):
    """Test get_optimal_vector raises error for invalid max_by_metric."""
    configured_simulation.run()
    with pytest.raises(ValueError):
        configured_simulation.get_optimal_vector(
            ("returns", "volatility", "sharpe"), "invalid"
        )


@pytest.mark.parametrize(
    "metric_keys",
    [
        ("returns", "volatility"),
        ("returns", "volatility", "sharpe", "extra"),
    ],
)
def test_get_optimal_vector_mismatched_metric_keys(configured_simulation, metric_keys):
    """Test get_optimal_vector raises error for mismatched metric_keys length."""
    configured_simulation.run()
    with pytest.raises(ValueError):
        configured_simulation.get_optimal_vector(metric_keys, metric_keys[0])
