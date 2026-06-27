from typing import Dict, Any
import numpy as np
import pandas as pd
from feature_extractors.features_factory import FeatureFactory
from workflows.abstract_workflow import AbstractTask


class ExtractFeaturesTask(AbstractTask):
    """TASK: Runs batch feature analysis and compiles the output Pandas summary table."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data_matrix: np.ndarray = context["data_matrix"]
        model_labels: np.ndarray = context["model_labels"]
        feature_names = context["feature_names"]

        total_realizations = data_matrix.shape[1]
        print(f"[TASK] Extracting analytical feature arrays across {total_realizations} data tracks...")

        # Batch operation over the completely integrated dataset matrix
        extracted_metrics = FeatureFactory.extract_batch(names=feature_names, data=data_matrix)

        # Build layout table strictly tracking metadata indexing and labels
        features = pd.DataFrame({
            "realization_idx": np.arange(total_realizations),
            "model_name": model_labels
        })

        # Append calculated analytical summary feature columns directly without mixing parameter data
        for feature_enum, metric_vector in extracted_metrics.items():
            features[feature_enum.value] = metric_vector

        context["features"] = features
        return context
