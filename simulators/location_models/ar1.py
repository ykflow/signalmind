import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName


class AR1Model(AbstractTimeSeriesModel):
    """Simulates multiple independent realizations of an AR(1) process with varying parameters."""

    def __init__(self, c: np.ndarray, phi: np.ndarray, sigma: np.ndarray):
        # Force inputs to be flat 1D arrays
        self.c = np.asarray(c).flatten()
        self.phi = np.asarray(phi).flatten()
        self.sigma = np.asarray(sigma).flatten()

        # Ensure all parameter arrays share the exact same length
        if not (self.c.shape == self.phi.shape == self.sigma.shape):
            raise ValueError("Arrays c, phi, and sigma must have identical dimensions.")

        # Check the stationarity condition across all parameters at once
        if np.any(np.abs(self.phi) >= 1.0):
            raise ValueError("All phi values must be strictly between -1 and 1 for stationarity.")

        # M represents the number of available parameter configurations
        self.M = self.c.shape[0]

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.AR1

    def simulate(self, num_steps: int, burn_in: int = 100) -> np.ndarray:
        if num_steps <= 0 or burn_in < 0:
            raise ValueError("Steps must be positive. Burn-in cannot be negative.")

        total_steps = burn_in + num_steps
        data = np.zeros((total_steps, self.M))
        eps = np.random.normal(0.0, self.sigma, size=(total_steps, self.M))

        # Vectorized generation across all M realizations simultaneously
        for t in range(1, total_steps):
            data[t, :] = self.c + self.phi * data[t - 1, :] + eps[t, :]

        return data[burn_in:, :]
