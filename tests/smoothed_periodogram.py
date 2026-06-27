import numpy as np
import matplotlib.pyplot as plt
from simulators.location_models.ar1 import AR1Model
from feature_extractors.features.spectral import PeriodogramExtractor


def theoretical_spectral_density(freqs: np.ndarray, phi: float, sigma: float = 1.0) -> np.ndarray:
    """Computes the exact analytical spectral density for an AR(1) process."""
    omega = freqs * np.pi
    return (sigma ** 2 / (2 * np.pi)) / (1 - 2 * phi * np.cos(omega) + phi ** 2)


def run_single_trace_comparison():
    # 1. Setup global configurations for 1 single realization
    num_steps = 1000  # Slightly longer time-series helps resolution
    num_realizations = 1  # Exactly 1 trace to see raw vs smooth behavior
    freqs_norm = np.linspace(0.001, 1, 300)

    # 2. Instantiate extractor engines
    raw_extractor = PeriodogramExtractor(target_frequencies=freqs_norm, smoothing_span=0)
    smooth_extractor = PeriodogramExtractor(target_frequencies=freqs_norm, smoothing_span=6)  # m=6 span

    # 3. Configure parameters for positive phi case (phi = 0.8)
    c_pos = np.zeros(num_realizations)
    phi_pos = np.ones(num_realizations) * 0.8
    sigma_pos = np.ones(num_realizations) * 1.0

    model_pos = AR1Model(c=c_pos, phi=phi_pos, sigma=sigma_pos)
    data_pos = model_pos.simulate(num_steps=num_steps, burn_in=300)

    # Extract 1D vectors out of the single realization column matrix
    raw_periodogram_pos = raw_extractor.extract(data_pos)[:, 0]
    smooth_periodogram_pos = smooth_extractor.extract(data_pos)[:, 0]
    theory_pos = theoretical_spectral_density(freqs_norm, phi=0.8)

    # 4. Configure parameters for negative phi case (phi = -0.8)
    c_neg = np.zeros(num_realizations)
    phi_neg = np.ones(num_realizations) * -0.8
    sigma_neg = np.ones(num_realizations) * 1.0

    model_neg = AR1Model(c=c_neg, phi=phi_neg, sigma=sigma_neg)
    data_neg = model_neg.simulate(num_steps=num_steps, burn_in=100)

    # Extract 1D vectors out of the single realization column matrix
    raw_periodogram_neg = raw_extractor.extract(data_neg)[:, 0]
    smooth_periodogram_neg = smooth_extractor.extract(data_neg)[:, 0]
    theory_neg = theoretical_spectral_density(freqs_norm, phi=-0.8)

    # 5. Visualization Generation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left Panel (phi = 0.8)
    ax1.plot(freqs_norm, raw_periodogram_pos, color='#1f77b4', alpha=0.35, linewidth=1, label='Raw Periodogram (Noisy)')
    ax1.plot(freqs_norm, smooth_periodogram_pos, color='#1f77b4', linewidth=2.5, label='Daniell Smoothed (m=6)')
    ax1.plot(freqs_norm, theory_pos, color='red', linestyle='--', linewidth=2, label='Analytical Curve')
    ax1.set_title(r'$\phi = 0.8$ (Single Realization)', fontsize=14, pad=10)
    ax1.set_xlabel(r'$\lambda / \pi$', fontsize=12)
    ax1.set_ylabel(r'$f_X(\lambda)$', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, np.max(theory_pos) * 1.5)
    ax1.legend(frameon=True, loc='upper right')
    ax1.grid(True, alpha=0.15)

    # Right Panel (phi = -0.8)
    ax2.plot(freqs_norm, raw_periodogram_neg, color='#1f77b4', alpha=0.35, linewidth=1, label='Raw Periodogram (Noisy)')
    ax2.plot(freqs_norm, smooth_periodogram_neg, color='#1f77b4', linewidth=2.5, label='Daniell Smoothed (m=6)')
    ax2.plot(freqs_norm, theory_neg, color='red', linestyle='--', linewidth=2, label='Analytical Curve')
    ax2.set_title(r'$\phi = -0.8$ (Single Realization)', fontsize=14, pad=10)
    ax2.set_xlabel(r'$\lambda / \pi$', fontsize=12)
    ax2.set_ylabel(r'$f_X(\lambda)$', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, np.max(theory_neg) * 1.5)
    ax2.legend(frameon=True, loc='upper right')
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_single_trace_comparison()
