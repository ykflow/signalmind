import unittest
import numpy as np
from simulators.location_models.ar1 import AR1Model
from feature_extractors.features.spectral import PeriodogramExtractor


class TestSpectralDensity(unittest.TestCase):

    def setUp(self):
        """Sets up baseline parameters for the simulations."""
        self.num_steps = 500
        self.num_realizations = 1500  # Higher count minimizes stochastic variance
        self.freqs_norm = np.linspace(0.005, 1, 100)
        self.extractor = PeriodogramExtractor(target_frequencies=self.freqs_norm)

    @staticmethod
    def _theoretical_spectral_density(freqs: np.ndarray, phi: float, sigma: float = 1.0) -> np.ndarray:
        """Computes analytical spectral density for an AR(1) process."""
        omega = freqs * np.pi
        return (sigma**2 / (2 * np.pi)) / (1 - 2 * phi * np.cos(omega) + phi**2)

    def _run_verification(self, phi_val: float):
        """Helper method to execute simulation and validate results against theory."""
        # Fix seed to ensure deterministic behavior across test runs
        np.random.seed(42)

        c = np.zeros(self.num_realizations)
        phi = np.ones(self.num_realizations) * phi_val
        sigma = np.ones(self.num_realizations) * 1.0

        # Simulate and extract empirical periodogram
        model = AR1Model(c=c, phi=phi, sigma=sigma)
        data = model.simulate(num_steps=self.num_steps, burn_in=100)
        empirical_periodogram = np.mean(self.extractor.extract(data), axis=1)

        # Get analytical truth
        theoretical_density = self._theoretical_spectral_density(self.freqs_norm, phi=phi_val)

        # 1. Assert shape correctness
        self.assertEqual(empirical_periodogram.shape, self.freqs_norm.shape)

        # 2. Assert value accuracy using Mean Absolute Error (MAE)
        mae = np.mean(np.abs(empirical_periodogram - theoretical_density))
        self.assertLess(
            mae,
            0.02,
            msg=f"Mean absolute error ({mae:.4f}) exceeded acceptable tolerance for phi={phi_val}"
        )

    def test_positive_phi_periodogram_matches_theory(self):
        """Validates empirical periodogram against analytical curve for phi = 0.8."""
        self._run_verification(phi_val=0.8)

    def test_negative_phi_periodogram_matches_theory(self):
        """Validates empirical periodogram against analytical curve for phi = -0.8."""
        self._run_verification(phi_val=-0.8)

    def test_periodogram_extractor_invalid_frequency_bounds(self):
        """Ensures that frequencies outside [0.0, 1.0] trigger a ValueError."""
        invalid_frequencies = np.array([-0.1, 0.5, 1.1])
        with self.assertRaisesRegex(ValueError, "Target frequencies must be between 0.0 and 1.0"):
            PeriodogramExtractor(target_frequencies=invalid_frequencies)


if __name__ == '__main__':
    unittest.main()
