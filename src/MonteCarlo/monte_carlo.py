from pydantic import BaseModel, conint
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np


class MonteCarloSettings(BaseModel):
    """Configuration settings for a Monte Carlo simulation.

    Stores parameters controlling the number of iterations, dimensions, and weight
    normalization behavior for the simulation.

    Attributes:
        iterations (int): Number of simulation trials to run. Must be positive.
        dimensions (int): Number of assets or variables in each trial. Must be positive.
        weight_normalisation (bool): If True, normalizes random weights to sum to 1.
    """

    iterations: conint(gt=0)
    dimensions: conint(gt=0)
    weight_normalisation: bool = False

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the random matrix for the simulation.

        Returns:
            Tuple[int, int]: A tuple of (iterations, dimensions) representing the shape
              of the random matrix generated for the simulation.
        """
        return self.iterations, self.dimensions

    def __str__(self) -> str:
        """Stylized string representation."""
        return f"""
            [Monte Carlo Simulation Settings]
            ------------------------------------------------------------
            Simulation Configuration:
                Iterations: {self.iterations}
                Dimensions: {self.dimensions}
                Weight Normalization: {'Enabled' if self.weight_normalisation else 'Disabled'}
            
            Random Matrix Shape:
                Shape: {self.shape}
            
            ------------------------------------------------------------"""

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"MonteCarloSettings(iterations={self.iterations}, "
            f"dimensions={self.dimensions}, weight_normalisation={self.weight_normalisation})"
        )


class MonteCarloSimulation:
    """A class for running Monte Carlo simulations with random weight vectors.

    Facilitates simulations by generating random weight vectors, applying a user-defined
    method to compute metrics, and storing results. Supports optimization to find the
    best trial based on a specified metric.

    Attributes:
        settings (MonteCarloSettings): Configuration settings for the simulation.
        simulation_func (Optional[Callable]): User-defined function to compute metrics for each trial.
        element_labels (Tuple[str] | None): Labels for the elements to allow mapping of a
            human-readable name to the results.
        metric_count (int | None): Number of metrics returned by the method.
        rng_cases (np.ndarray | None): Matrix of random weight vectors, shape (iterations, dimensions).
        simulation_inputs (Dict | None): Keyword arguments for the method.
        results (np.ndarray | None): Array of simulation results, shape (iterations, metric_count).
    """

    def __init__(self, settings: MonteCarloSettings):
        self.settings = settings

        self.simulation_func: Optional[Callable] = None
        self.simulation_inputs = None
        self.element_labels: Tuple[str, ...] | None = None
        self.metric_count: int | None = None
        self.random_trials = None
        self.results = None

    def set_simulation_function(self, simulation_func: Callable, metric_count: int):
        """Loads the simulation method and its output size.

        Args:
            simulation_func (Callable): Callable that computes metrics for a random weight vector.
                Must accept a weight vector (np.ndarray) in the first position; subsequent
                keyword arguments can then be specified, returning a NumPy array of shape (metric_count,).
            metric_count (int): Number of metrics returned by the function.
        """
        if not metric_count > 0:
            raise ValueError(f"metric_count ({metric_count}) must be greater than 0!")

        self.simulation_func = simulation_func
        self.metric_count = metric_count

    def set_simulation_inputs(self, element_labels: Tuple[str, ...], **input_data):
        """Sets the input data for the simulation function.

        Configures the simulation by specifying the keyword arguments
        to be passed to the simulation function during execution.

        Args:
            element_labels Tuple[str, ...]: Labels for the elements to allow mapping
                of a human-readable name to the results.
            **input_data (Dict): Arbitrary keyword arguments to be passed to the simulation
                function.
        """
        self.element_labels = element_labels
        self.simulation_inputs = input_data

    def _generate_random_matrix(self) -> np.ndarray:
        """Generates a matrix of random weight vectors.

        Creates a matrix of shape (iterations, dimensions) with random values uniformly
        distributed in [0, 1). If weight normalization is enabled, each row is normalized
        to sum to 1.

        Returns:
            rng_matrix (np.ndarray): Random weight matrix, shape (iterations, dimensions). If
                weight_normalisation is True, each row sums to 1.

        Raises:
            ValueError: If settings.shape contains invalid dimensions (e.g., negative or zero).
        """
        rng = np.random.default_rng()
        rng_matrix = rng.random(size=self.settings.shape)
        if self.settings.weight_normalisation:
            return rng_matrix / rng_matrix.sum(axis=1).reshape(-1, 1)
        return rng_matrix

    def run(self):
        """Runs the Monte Carlo simulation.

        Generates random weight vectors, applies the loaded function to each vector, and
        stores the results in a matrix of shape (iterations, metric_count).

        Returns:
            np.ndarray: Simulation results, shape (iterations, metric_count), where each
                row contains the metrics for one trial.

        Raises:
            ValueError: If simulation_func, input_data, or metric_count are not set.
            TypeError: If the simulation_func returns an incompatible output type or shape.
        """
        if self.simulation_func is None:
            raise ValueError(
                "No simulation function specified. Call set_simulation_function first."
            )
        if self.simulation_inputs is None:
            raise ValueError(
                "No simulation inputs specified. Call set_simulation_inputs first."
            )

        self.random_trials = self._generate_random_matrix()

        results_shape = (self.settings.iterations, self.metric_count)
        self.results = np.zeros(results_shape)

        for i in range(self.settings.iterations):
            random_vector = self.random_trials[i, :]
            self.results[i] = self.simulation_func(
                random_vector, **self.simulation_inputs
            )
        return self.results

    def get_optimal_vector(
        self, metric_keys: Tuple[str, ...], max_by_metric: str
    ) -> Dict[str, Any]:
        """Returns the optimal trial based on maximizing a specified metric.

        Identifies the trial that maximizes the metric specified by max_by_key, returning
        a dictionary with the corresponding weights and metric values.

        Args:
            metric_keys (Tuple[str]): Tuple of strings naming the metrics in results
                columns.
            max_by_metric (str): Metric name to maximize. Must be in metric_keys.

        Returns:
            Dict: Dictionary containing:
                * case (np.ndarray): Optimal weight vector, shape (dimensions,).
                * metric_keys (float): Metric values for the optimal trial, in the order
                    of metric_keys.

        Raises:
            ValueError: If results are not generated, max_by_key is not in metric_keys,
                or metric_keys length does not match results columns.
        """
        if self.results is None:
            ValueError("First run the simulation to generate results.")

        if max_by_metric not in metric_keys:
            raise ValueError(
                f"max_by '{max_by_metric}' must be in metric_keys: {metric_keys}"
            )

        if len(metric_keys) != self.results.shape[1]:
            raise ValueError(
                f"metric_keys length ({len(metric_keys)})"
                f"should match the output shape of the results ({self.results.shape[1]})"
            )

        optimal_metric_idx = metric_keys.index(max_by_metric)

        optimal_result_idx = np.argmax(self.results[:, optimal_metric_idx])

        optimal_result_labelled = {
            name: value
            for name, value in zip(
                self.element_labels, self.random_trials[optimal_result_idx]
            )
        }

        optimal_result = {"case": optimal_result_labelled}
        for i, _key in enumerate(metric_keys):
            optimal_result[_key] = self.results[optimal_result_idx, i]
        return optimal_result
