import numpy as np
import pandas as pd
from feature_extractors.abstract_features import BaseFeatureExtractor
from feature_extractors.enum_features import FeatureDomain, FeatureName


class MeanExtractor(BaseFeatureExtractor):
    """Computes empirical mean across realization paths."""

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.STATISTICAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.MEAN

    def extract(self, data: np.ndarray) -> pd.DataFrame:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")
        return pd.DataFrame(np.mean(data, axis=0), columns=[self.feature_name.value])


class VarianceExtractor(BaseFeatureExtractor):
    """Computes empirical variance across realization paths."""

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.STATISTICAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.VARIANCE

    def extract(self, data: np.ndarray) -> pd.DataFrame:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")
        return pd.DataFrame(np.var(data, axis=0), columns=[self.feature_name.value])


class SkewnessExtractor(BaseFeatureExtractor):
    """Computes Fisher-Pearson standardized skewness across realization paths."""

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.STATISTICAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.SKEWNESS

    def extract(self, data: np.ndarray) -> pd.DataFrame:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0.0, 1e-12, std)  # Prevent division by zero if series is perfectly flat
        m3 = np.mean((data - mean) ** 3, axis=0)
        skew = (m3 / (std ** 3))
        return pd.DataFrame(skew, columns=[self.feature_name.value])


class KurtosisExtractor(BaseFeatureExtractor):
    """Computes excess kurtosis across realization paths (Normal Distribution = 0.0)."""

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.STATISTICAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.KURTOSIS

    def extract(self, data: np.ndarray) -> pd.DataFrame:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")

        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)
        var = np.where(var == 0.0, 1e-12, var) # Prevent division by zero
        m4 = np.mean((data - mean) ** 4, axis=0)
        kurt = (m4 / (var ** 2)) - 3.0
        return pd.DataFrame(kurt, columns=[self.feature_name.value])
