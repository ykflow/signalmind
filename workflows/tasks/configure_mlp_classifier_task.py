from typing import Any, Dict
from workflows.abstract_workflow import AbstractTask
from utilities.input_output_dimension_calculator import InputOutputDimensionsCalculator
from neural.optimizer_factory import OptimizerFactory
from neural.set_layers_mlp_classifier import SetLayersMlpClassifier


class ConfigureMlpClassifierTask(AbstractTask):
    """TASK: Configures SignalMind's MLP Classifier."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = context["config"]
        metadata = InputOutputDimensionsCalculator(config)
        model = SetLayersMlpClassifier(config, metadata)
        optimizer = OptimizerFactory(config).create(model)
        context["neural"] = dict({"metadata": metadata, "model": model, "optimizer": optimizer})
        return context
