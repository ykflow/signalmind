import pandas as pd
import numpy as np
from feature_extractors.abstract_features import BaseFeatureExtractor
from feature_extractors.enum_features import FeatureDomain, FeatureName


class TrendSlopeExtractor(BaseFeatureExtractor):
    """Computes the linear trend slope across realization paths using OLS."""

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.STRUCTURAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.TREND_SLOPE

    def extract(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")

        num_steps, num_realizations = data.shape
        if num_steps < 2:
            raise ValueError("Time series length must be at least 2 steps to calculate a trend line.")

        t = np.arange(num_steps)
        mean_t = (num_steps - 1) / 2.0  # Analytical mean of arange(num_steps)
        mean_data = np.mean(data, axis=0)  # Shape: (num_realizations,)
        t_centered = t[:, np.newaxis] - mean_t
        var_t = np.sum(t_centered ** 2)
        covariance_t_data = np.sum(t_centered * (data - mean_data), axis=0)
        slope = covariance_t_data / var_t  # slope (beta) = Cov(time, Y) / Var(time)
        return pd.DataFrame(slope, columns=[self.feature_name.value])


class HurstExponentExtractor(BaseFeatureExtractor):
    """Estimates the Hurst Exponent using a vectorized Rescaled Range (R/S) algorithm."""

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.TEMPORAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.HURST_EXPONENT

    def extract(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")

        num_steps, num_realizations = data.shape
        if num_steps < 32:
            raise ValueError("Time series length must be at least 32 steps to calculate Hurst Exponent.")

        max_lag = int(np.floor(num_steps / 2))
        lags = np.unique(np.geomspace(10, max_lag, num=10).astype(int))

        log_rs_results = np.zeros((len(lags), num_realizations))
        for idx, lag in enumerate(lags):             # Shape: (num_blocks, lag, num_realizations)
            num_blocks = num_steps // lag
            truncated_len = num_blocks * lag
            blocks = data[:truncated_len, :].reshape(num_blocks, lag, num_realizations)
            block_means = np.mean(blocks, axis=1, keepdims=True)
            centered_blocks = blocks - block_means  # Local mean centering per block chunk

            cum_deviations = np.cumsum(centered_blocks, axis=1)
            r_val = np.max(cum_deviations, axis=1) - np.min(cum_deviations, axis=1)
            s_val = np.std(blocks, axis=1)

            s_val = np.where(s_val == 0.0, 1e-12, s_val) # Guard against structural division by zero
            rs_vector = np.mean(r_val / s_val, axis=0)  # Average the rescaled ranges (R/S) across all available chunks
            log_rs_results[idx, :] = np.log(np.where(rs_vector <= 0.0, 1e-12, rs_vector))

        log_lags = np.log(lags)
        hurst_values = np.zeros(num_realizations)
        for m in range(num_realizations):
            slope, _ = np.polyfit(log_lags, log_rs_results[:, m], 1) # OLS step (slope represents the Hurst Exponent)
            hurst_values[m] = slope

        return pd.DataFrame(hurst_values, columns=[self.feature_name.value])
