from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


@dataclass
class MonteCarloSettings:
    iterations: int
    dimensions: int
    weight_normalisation: bool = False

    @property
    def shape(self) -> Tuple[int, int]:
        return self.iterations, self.dimensions


class MonteCarloSimulation:

    def __init__(self, settings: MonteCarloSettings):
        self.settings = settings

        self.method: Optional[Callable] = None
        self.random_generator = None
        self.rng_cases = None
        self.datasets = None
        self.results = None

    def load_method(self, callable: Callable, output_size: int):
        """Takes the abstract logic and loads it for the simulation."""
        self.method = callable
        self.output_size = output_size

    def load_with_parameters(self, **kwargs):
        """Takes abstract keyword arguments and loads them for the simulation."""
        self.datasets = kwargs

    def _generate_random_matrix(self) -> np.ndarray:
        """Sets up the random generator."""
        rng = np.random.default_rng()
        rng_matrix = rng.random(size=self.settings.shape)
        if self.settings.weight_normalisation:
            return rng_matrix / rng_matrix.sum(axis=1).reshape(-1, 1)
        return rng_matrix

    def run(self):
        if self.method is None:
            raise ValueError("No method loaded. Call load_method first.")
        if self.datasets is None:
            raise ValueError("No datasets loaded. Call load_datasets first.")

        self.rng_cases = self._generate_random_matrix()

        results_shape = (self.settings.iterations, self.output_size)
        self.results = np.zeros(results_shape)

        for i in range(self.settings.iterations):
            random_vector = self.rng_cases[i, :]
            self.results[i] = self.method(random_vector, **self.datasets)
        return self.results

    def get_optimal_vector(self, metric_keys: Tuple[str], max_by_key: str):
        """
        Returns the optimal vector information based on the provided metrics.
        The first metric in the keys will be used to determine 'optimal'.
        """
        if self.results is None:
            ValueError('First run the simulation to generate results.')

        if max_by_key not in metric_keys:
            raise ValueError(f"max_by '{max_by_key}' must be in metric_keys: {metric_keys}")

        if len(metric_keys) != self.results.shape[1]:
            raise ValueError(f'metric_keys length ({len(metric_keys)})'
                             f'should match the output shape of the results ({self.results.shape[1]})')

        optimal_metric_idx = metric_keys.index(max_by_key)
        idx = np.argmax(self.results[:, optimal_metric_idx])
        optimal_result = {'case': self.rng_cases[idx]}
        for i, _key in enumerate(metric_keys):
            optimal_result[_key] = self.results[idx, i]
        return optimal_result
