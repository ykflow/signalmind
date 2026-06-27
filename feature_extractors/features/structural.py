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
        return covariance_t_data / var_t  # slope (beta) = Cov(time, Y) / Var(time)


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

        # Accumulate log(R/S) averages per lag interval
        log_rs_results = np.zeros((len(lags), num_realizations))

        # 2. Vectorized loop over lag configurations
        for idx, lag in enumerate(lags):
            # Shape: (num_blocks, lag, num_realizations)
            num_blocks = num_steps // lag
            truncated_len = num_blocks * lag

            # Reshape into split segments to process blocks simultaneously
            blocks = data[:truncated_len, :].reshape(num_blocks, lag, num_realizations)

            # Local mean centering per block chunk
            block_means = np.mean(blocks, axis=1, keepdims=True)
            centered_blocks = blocks - block_means

            # Cumulative deviations inside chunks
            cum_deviations = np.cumsum(centered_blocks, axis=1)

            # Calculate range (R) and standard deviations (S) per block across columns
            r_val = np.max(cum_deviations, axis=1) - np.min(cum_deviations, axis=1)
            s_val = np.std(blocks, axis=1)

            # Guard against structural division by zero
            s_val = np.where(s_val == 0.0, 1e-12, s_val)

            # Average the rescaled ranges (R/S) across all available chunks
            rs_vector = np.mean(r_val / s_val, axis=0)
            log_rs_results[idx, :] = np.log(np.where(rs_vector <= 0.0, 1e-12, rs_vector))

        # 3. Fit a line (log(lag) vs log(R/S)) for each realization column
        log_lags = np.log(lags)
        hurst_values = np.zeros(num_realizations)

        for m in range(num_realizations):
            # Linear least-squares fit (slope represents the Hurst Exponent)
            slope, _ = np.polyfit(log_lags, log_rs_results[:, m], 1)
            hurst_values[m] = slope

        return hurst_values
