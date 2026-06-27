from enum import Enum

class FeatureDomain(Enum):
    STATISTICAL = "Statistical"
    STRUCTURAL = "Structural"
    TEMPORAL = "Temporal"


class FeatureName(Enum):
    MEAN = "mean"
    VARIANCE = "variance"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    HURST_EXPONENT = "hurst_exponent"
    TREND_SLOPE = "trend_slope"
    SPECTRAL_DENSITY = "spectral_density"
