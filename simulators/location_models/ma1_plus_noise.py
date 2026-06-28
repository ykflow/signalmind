from typing import Any
import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ModelType, ProcessName
from simulators.location_models.ma1 import MA1Model
from simulators.location_models.white_noise import WhiteNoiseModel

class MA1NoiseModel(AbstractTimeSeriesModel):
    """Simulates multiple independent realizations of an MA(1) process observed with measurement noise."""

    def __init__(self, c: np.ndarray, theta: np.ndarray, sigma_state: np.ndarray, sigma_measurement: np.ndarray,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self._state_model = MA1Model(c=c, theta=theta, sigma=sigma_state)
        self._noise_model = WhiteNoiseModel(c=np.zeros_like(c), sigma=sigma_measurement)

        # Enforce that all input arrays map to the exact same dimension M
        if self._state_model.M != self._noise_model.M:
            raise ValueError("State (MA1) and Noise (WN) parameter dimensions must match.")

        self.M = self._state_model.M

    @property
    def model_type(self) -> ModelType:
        return ModelType.LOCATION

    @property
    def process_name(self) -> ProcessName:
        return ProcessName.MA1_NOISE

    def simulate(self, num_steps: int,  burn_in: int=250) -> np.ndarray:
        latent_states = self._state_model.simulate(num_steps, burn_in)
        measurement_errors = self._noise_model.simulate(num_steps, burn_in)
        data = latent_states + measurement_errors
        return data
