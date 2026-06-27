import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName
from simulators.location_models.ar1 import AR1Model
from simulators.location_models.white_noise import WhiteNoiseModel


class AR1NoiseModel(AbstractTimeSeriesModel):
    def __init__(self, c: np.ndarray, phi: np.ndarray, sigma_state: np.ndarray, sigma_noise: np.ndarray):
        self._state_model = AR1Model(c=c, phi=phi, sigma=sigma_state)
        self._noise_model = WhiteNoiseModel(c=np.zeros_like(c), sigma=sigma_noise)

        if self._state_model.M != self._noise_model.M:
            raise ValueError("State and Noise parameter dimensions must match.")
        self.M = self._state_model.M

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.AR1_NOISE

    def simulate(self, num_steps: int, burn_in: int = 100) -> np.ndarray:
        latent_states = self._state_model.simulate(num_steps, burn_in)
        measurement_errors = self._noise_model.simulate(num_steps, burn_in)
        data = latent_states + measurement_errors
        return data
