import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName

class RandomWalkModel(AbstractTimeSeriesModel):
    """Simulates multiple independent realizations of a driftless Random Walk."""

    def __init__(self, sigma: np.ndarray):
        self.sigma = np.asarray(sigma).flatten()
        self.M = self.sigma.shape

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.RANDOM_WALK

    def simulate(self, num_steps: int, burn_in: int = 100) -> np.ndarray:
        if num_steps <= 0 or burn_in < 0:
            raise ValueError("Steps must be positive. Burn-in cannot be negative.")

        total_steps = burn_in + num_steps
        eps = np.random.normal(0.0, self.sigma, size=(total_steps, self.M))
        data = np.cumsum(eps, axis=0)

        return data[burn_in:, :]
