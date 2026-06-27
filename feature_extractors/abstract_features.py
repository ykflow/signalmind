from abc import ABC, abstractmethod
import numpy as np
from feature_extractors.enum_features import FeatureDomain, FeatureName


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extraction modules."""

    @property
    @abstractmethod
    def domain(self) -> FeatureDomain:
        ...

    @property
    @abstractmethod
    def feature_name(self) -> FeatureName:
        ...

    @abstractmethod
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extracts features across a 2D matrix of shape (num_steps, num_realizations).

        Returns:
            A 1D numpy array of shape (num_realizations,) containing the extracted feature.
        """
        ...
