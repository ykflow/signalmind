from typing import Any
import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName


class RandomWalkDriftModel(AbstractTimeSeriesModel):
    """Simulates multiple independent realizations of a Random Walk with drift."""

    def __init__(self, drift: np.ndarray, sigma: np.ndarray, **kwargs: Any):
        super().__init__(**kwargs)
        self.drift = np.asarray(drift).flatten()
        self.sigma = np.asarray(sigma).flatten()

        if self.drift.shape != self.sigma.shape:
            raise ValueError("Arrays drift and sigma must have identical dimensions.")

        if np.any(self.sigma <= 0.0):
            raise ValueError("All sigma values must be strictly greater than 0.")

        self.M = self.sigma.shape[0]

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.RANDOM_WALK_DRIFT

    def simulate(self, num_steps: int, burn_in: int = 100) -> np.ndarray:
        if num_steps <= 0 or burn_in < 0:
            raise ValueError("Steps must be positive. Burn-in cannot be negative.")

        total_steps = burn_in + num_steps
        t = np.arange(total_steps)[:, np.newaxis]
        eps = np.random.normal(0.0, self.sigma, size=(total_steps, self.M))
        stochastic_component = np.cumsum(eps, axis=0)
        deterministic_trend = t * self.drift
        data = deterministic_trend + stochastic_component
        return data[burn_in:, :]
