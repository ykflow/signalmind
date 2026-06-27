from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from simulators.enum_models import ModelType, ProcessName


class AbstractTimeSeriesModel(ABC):
    """Abstract base class for all simulation models."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Returns whether the model dictates mean or variance."""
        ...

    @property
    @abstractmethod
    def process_name(self) -> ProcessName:
        """Returns the specific statistical process name."""
        ...

    @abstractmethod
    def simulate(self, num_steps: int, burn_in: int=250) -> np.ndarray:
        """Generates a synthetic time-series trajectory."""
        ...
