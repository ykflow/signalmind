from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd

from feature_extractors.enum_features import FeatureDomain, FeatureName


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extraction modules."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    @abstractmethod
    def domain(self) -> FeatureDomain:
        ...

    @property
    @abstractmethod
    def feature_name(self) -> FeatureName:
        ...

    @abstractmethod
    def extract(self, data: np.ndarray) -> pd.DataFrame:
        """Extracts features across a pandas DataFrame of shape (num_steps, num_realizations).

        Returns:
            A 1D numpy array of shape (num_realizations,) containing the extracted feature.
        """
        ...
