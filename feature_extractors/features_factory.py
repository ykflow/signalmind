from typing import Dict, Type, List
import numpy as np
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
    def get_extractor(cls, name: FeatureName) -> BaseFeatureExtractor:
        extractor_class = cls._registry.get(name)
        if not extractor_class:
            raise ValueError(f"No feature extractor registered for name: {name}")
        return extractor_class()

    @classmethod
    def extract_batch(cls, names: List[FeatureName], data: np.ndarray) -> Dict[FeatureName, np.ndarray]:
        results = {}
        for name in names:
            extractor = cls.get_extractor(name)
            results[name] = extractor.extract(data)
        return results
