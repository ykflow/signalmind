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
        print(f"[TASK] Extracting analytical feature arrays for {total_realizations} time-series...")
        extracted_metrics = FeatureFactory.extract_batch(feature_configs=feature_names, data=data_matrix)
        features = pd.concat(extracted_metrics, axis=1)
        features = features.set_index(model_labels)
        context["features"] = features
        return context
