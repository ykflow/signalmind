import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName


class WhiteNoiseModel(AbstractTimeSeriesModel):
    """Simulates multiple independent realizations of a White Noise process."""

    def __init__(self, c: np.ndarray, sigma: np.ndarray):
        self.c = np.asarray(c).flatten()
        self.sigma = np.asarray(sigma).flatten()

        if self.c.shape != self.sigma.shape:
            raise ValueError("Arrays c and sigma must have identical dimensions.")

        self.M = self.c.shape[0]

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.WHITE_NOISE

    def simulate(self, num_steps: int, burn_in: int=250) -> np.ndarray:
        if num_steps <= 0 or burn_in < 0:
            raise ValueError("Steps must be positive. Burn-in cannot be negative.")

        total_steps = burn_in + num_steps
        eps = np.random.normal(0.0, self.sigma, size=(total_steps, self.M))
        data = self.c + eps  # unstandardized WN
        return data[burn_in:, :]