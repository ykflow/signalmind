from typing import Dict, Type, Any
import numpy as np
import pandas as pd

from feature_extractors.enum_features import FeatureName
from feature_extractors.abstract_features import BaseFeatureExtractor
from feature_extractors.features.statistical import (
    MeanExtractor,
    VarianceExtractor,
    SkewnessExtractor,
    KurtosisExtractor,
)
from feature_extractors.features.structural import HurstExponentExtractor, TrendSlopeExtractor
from feature_extractors.features.spectral import PeriodogramExtractor


class FeatureFactory:
    """Factory class to register, instantiate, and execute feature extractors dynamically."""
    _registry: Dict[FeatureName, Type[BaseFeatureExtractor]] = {
        FeatureName.MEAN: MeanExtractor,
        FeatureName.VARIANCE: VarianceExtractor,
        FeatureName.SKEWNESS: SkewnessExtractor,
        FeatureName.KURTOSIS: KurtosisExtractor,
        FeatureName.HURST_EXPONENT: HurstExponentExtractor,
        FeatureName.TREND_SLOPE: TrendSlopeExtractor,
        FeatureName.SPECTRAL_DENSITY: PeriodogramExtractor,
    }

    @classmethod
    def get_extractor(cls, name: FeatureName, **kwargs: Any) -> BaseFeatureExtractor:
        """Instantiates extractor. Passes optional configurations via **kwargs."""
        extractor_class = cls._registry.get(name)
        if not extractor_class:
            raise ValueError(f"No feature extractor registered for name: {name}")
        return extractor_class(**kwargs)

    @classmethod
    def extract_batch(cls, feature_configs: Dict[FeatureName, Dict[str, Any]],data: np.ndarray) -> list[pd.DataFrame]:
        """
        Executes a batch of features.

        feature_configs: Dict mapping FeatureName -> parameter dictionary (e.g., {'target_frequencies': ...})
        """
        results = []
        for name, params in feature_configs.items():
            extractor = cls.get_extractor(name, **params)
            results.append(extractor.extract(data))
        return results
