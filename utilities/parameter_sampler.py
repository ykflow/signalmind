from typing import Dict
import numpy as np
from utilities.config_reader import ModelConfig


class ParameterSampler:
    """Handles generation of randomized parameter vectors from configuration bounds."""

    @classmethod
    def sample_uniform(cls, model_cfg: ModelConfig, num_realizations: int) -> Dict[str, np.ndarray]:
        """Generates flat 1D numpy arrays uniformly sampled across model configuration boundaries.

        Args:
            model_cfg: The parsed ModelConfig containing parameter name bounds.
            num_realizations: Dimension M (number of concurrent simulation paths).

        Returns:
            A dictionary mapping parameter names to 1D arrays of shape (num_realizations,).
        """
        sampled_vectors = {}
        for param, bounds in model_cfg.parameter_bounds.items():
            low, high = bounds

            # Defensive check: ensure the configuration bounds are ordered correctly
            if low > high:
                raise ValueError(
                    f"Invalid boundary condition for parameter '{param}' in model '{model_cfg.name}': "
                    f"lower bound ({low}) cannot be greater than upper bound ({high})."
                )

            sampled_vectors[param] = np.random.uniform(low, high, size=num_realizations)

        return sampled_vectors
