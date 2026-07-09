from feature_extractors.enum_features import FeatureName


class InputOutputDimensionsCalculator:
    """
    Parses and manages structural settings, input features, and target dimensions
    """

    def __init__(self, config: dict):
        self.config = config
        self.input_dim = self._compute_input_dimension()
        self.classes = self._compute_active_classes()
        self.num_classes = len(self.classes)

        # Bi-directional maps for converting integer labels back to text names
        self.label_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_label = {idx: name for idx, name in enumerate(self.classes)}

    def _compute_input_dimension(self) -> int:
        """Calculates input vector dimension based on enabled features."""
        dim = 0
        for feature in self.config.get('features', []):
            if not feature.get('enabled', False):
                continue

            # Special case handling for vector features like spectral densities
            if feature.get('name') == FeatureName.SPECTRAL_DENSITY:
                freqs = feature.get('parameters').get('target_frequencies')
                dim += len(freqs)
            else:
                dim += 1  # Basic scalar statistical markers (Mean, Var, etc.)
        return dim

    def _compute_active_classes(self) -> list:
        """Extracts names of all target models marked enabled."""
        return [ model['name'] for model in self.config.get('models', [])  if model.get('enabled', False)]