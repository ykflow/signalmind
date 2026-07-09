from typing import Any, Dict
import numpy as np
from workflows.abstract_workflow import AbstractTask
from utilities.parameter_sampler import ParameterSampler


class GenerateModelParameterTask(AbstractTask):
    """TASK: Simulates individual parameter arrays for every active model profile."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = context["config"]
        M = context["num_realizations"]
        model_parameter_grids: Dict[str, Dict[str, np.ndarray]] = {}

        print(f"[TASK] Simulating parameter grids from configuration boundaries (M={M})...")
        for model_cfg in config.models:
            if not model_cfg.enabled:
                continue

            print(f" -> Generating uniform parameter vectors for: {model_cfg.name}")
            param_vectors = ParameterSampler.sample_uniform(model_cfg=model_cfg, num_realizations=M)
            model_parameter_grids[model_cfg.name] = param_vectors

        context["model_parameter_grids"] = model_parameter_grids
        return context
