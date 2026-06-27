import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName
from simulators.location_models.random_walk import RandomWalkModel
from simulators.location_models.white_noise import WhiteNoiseModel


class RandomWalkNoiseModel(AbstractTimeSeriesModel):
    """Simulates a driftless Random Walk process observed with measurement noise."""

    def __init__(self, sigma_state: np.ndarray, sigma_noise: np.ndarray):
        self._state_model = RandomWalkModel(sigma=sigma_state)
        self._noise_model = WhiteNoiseModel(c=np.zeros_like(sigma_state), sigma=sigma_noise)
        self.M = self._state_model.M

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.RANDOM_WALK_NOISE

    def simulate(self, num_steps: int, burn_in: int = 100) -> np.ndarray:
        latent_states = self._state_model.simulate(num_steps=num_steps, burn_in=burn_in)
        measurement_errors = self._noise_model.simulate(num_steps=num_steps, burn_in=burn_in)
        combined = latent_states + measurement_errors
        return combined[burn_in:, :]
