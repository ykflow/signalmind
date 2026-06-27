import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName


class MA1Model(AbstractTimeSeriesModel):
    """Simulates multiple independent realizations of an MA(1) process."""

    def __init__(self, c: np.ndarray, theta: np.ndarray, sigma: np.ndarray):
        self.c = np.asarray(c).flatten()
        self.theta = np.asarray(theta).flatten()
        self.sigma = np.asarray(sigma).flatten()

        if not (self.c.shape == self.theta.shape == self.sigma.shape):
            raise ValueError("Arrays c, theta, and sigma must have identical dimensions.")

        #invertibality check: np.any(np.abs(self.theta) >= 1)
        if np.any(np.abs(self.theta) >= 1.0):
            raise ValueError("All theta values must be strictly between -1 and 1 for stationarity.")

        self.M = self.c.shape[0]

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.MA1

    def simulate(self, num_steps: int, burn_in: int = 100) -> np.ndarray:
        if num_steps <= 0 or burn_in < 0:
            raise ValueError("Steps must be positive. Burn-in cannot be negative.")

        total_steps = burn_in + num_steps
        eps = np.random.normal(0.0, self.sigma, size=(total_steps, self.M))
        data = np.zeros((total_steps, self.M))

        # Vectorized MA(1) step: data_t = c + theta * eps_{t-1} + eps_t
        # Since it depends only on the noise sequence, we can fully vectorize without a loop
        data[0, :] = self.c + eps[0, :]  # Assumes eps[-1] is 0
        data[1:, :] = self.c + self.theta * eps[:-1, :] + eps[1:, :]

        return data[burn_in:, :]
