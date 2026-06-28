from typing import Any
import numpy as np
import pandas as pd

from feature_extractors.abstract_features import BaseFeatureExtractor
from feature_extractors.enum_features import FeatureDomain, FeatureName


class PeriodogramExtractor(BaseFeatureExtractor):
    """Computes the mathematically exact Spectral Density Periodogram at any frequency with optional Daniell smoothing."""

    def __init__(self, target_frequencies: np.ndarray, smoothing_span: int = 0, **kwargs: Any):
        """
        Args:
            target_frequencies: 1D array of normalized frequencies to extract,
                                scaled by pi (0.0 means lambda=0, 1.0 means lambda=pi).
            smoothing_span: Half-bandwidth parameter 'm' for a Modified Daniell Filter.
                            0 means no smoothing. m > 0 looks m steps left and right.
        """
        super().__init__(**kwargs)
        self._target_frequencies = np.asarray(target_frequencies).flatten()
        if np.any(self._target_frequencies < 0.0) or np.any(self._target_frequencies > 1.0):
            raise ValueError("Target frequencies must be between 0.0 and 1.0 (scaled by pi).")

        if smoothing_span < 0:
            raise ValueError("Smoothing span 'm' cannot be negative.")
        self.smoothing_span = smoothing_span

    @property
    def domain(self) -> FeatureDomain:
        return FeatureDomain.TEMPORAL

    @property
    def feature_name(self) -> FeatureName:
        return FeatureName.SPECTRAL_DENSITY

    def _apply_daniell_kernel(self, raw_periodogram: np.ndarray) -> np.ndarray:
        """Applies a Modified Daniell filter across the frequency rows (axis=0) using reflection."""
        k = self.smoothing_span
        if k == 0:
            return raw_periodogram

        weights = np.ones(2 * k + 1) / (2.0 * k)
        weights[0] = 1.0 / (4.0 * k)
        weights[-1] = 1.0 / (4.0 * k)

        # Use 'reflect' mode so neighboring frequencies are mirrored
        # past the boundaries instead of duplicating the artificial zero at index 0.
        padded = np.pad(raw_periodogram, ((k, k), (0, 0)), mode='reflect')

        # convolute over frequency rows
        smoothed = np.apply_along_axis( lambda col: np.convolve(col, weights, mode='valid'), axis=0, arr=padded)
        return smoothed

    def extract(self, data: np.ndarray) -> pd.DataFrame:
        if data.ndim != 2:
            raise ValueError("Input data matrix must be 2D: (num_steps, num_realizations)")

        num_steps, M = data.shape
        centered_data = data - np.mean(data, axis=0)

        t_vec = np.arange(1, num_steps + 1).reshape(num_steps, 1)  # Shape: (num_steps, 1)
        omega_vec = self._target_frequencies * np.pi  # Shape: (num_frequencies,)
        kernel_matrix = np.exp(-1j * t_vec * omega_vec)
        dft_sum = np.matmul(kernel_matrix.T, centered_data)

        exact_periodogram = (np.abs(dft_sum) ** 2) / (2.0 * np.pi * num_steps)
        result = self._apply_daniell_kernel(exact_periodogram).T
        result = pd.DataFrame(result, columns = [f'{FeatureName.SPECTRAL_DENSITY.value}-{x}' for x in self._target_frequencies])
        return result
