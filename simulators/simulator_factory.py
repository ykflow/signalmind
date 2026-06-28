from typing import Dict, Type
import numpy as np
from simulators.abstract_models import AbstractTimeSeriesModel
from simulators.enum_models import ProcessName
from simulators.location_models.ar1 import AR1Model
from simulators.location_models.ar1_plus_noise import AR1NoiseModel
from simulators.location_models.ma1 import MA1Model
from simulators.location_models.ma1_plus_noise import MA1NoiseModel
from simulators.location_models.random_walk import RandomWalkModel
from simulators.location_models.random_walk_drift import RandomWalkDriftModel
from simulators.location_models.random_walk_noise import RandomWalkNoiseModel
from simulators.location_models.white_noise import WhiteNoiseModel


class SimulatorFactory:
    """Central registry and creator engine for all time series simulation models."""
    _registry: Dict[ProcessName, Type[AbstractTimeSeriesModel]] = {
        ProcessName.WHITE_NOISE: WhiteNoiseModel,
        ProcessName.AR1: AR1Model,
        ProcessName.MA1: MA1Model,
        ProcessName.AR1_NOISE: AR1NoiseModel,
        ProcessName.MA1_NOISE: MA1NoiseModel,
        ProcessName.RANDOM_WALK: RandomWalkModel,
        ProcessName.RANDOM_WALK_DRIFT: RandomWalkDriftModel,
        ProcessName.RANDOM_WALK_NOISE: RandomWalkNoiseModel,
    }

    @classmethod
    def create(cls, process_name: ProcessName, parameters: Dict[str, np.ndarray]) -> AbstractTimeSeriesModel:
        model_class = cls._registry.get(process_name)
        if not model_class:
            raise ValueError(
                f"No simulation model class has been registered for process: '{process_name.value}'. "
                f"Ensure it is appended to SimulatorFactory._registry."
            )

        try:
            return model_class(**parameters)
        except TypeError as e:
            raise TypeError(
                f"Parameter mismatch while instantiating model class '{model_class.__name__}' "
                f"for process '{process_name.value}': {e}"
            )