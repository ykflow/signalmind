from typing import Any, Dict, List
import numpy as np
from workflows.abstract_workflow import AbstractTask
from simulators.simulator_factory import SimulatorFactory


class SimulateModelsTask(AbstractTask):
    """TASK: Instantiates registered factory models and runs vectorized data simulations."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = context["config"]
        M = context["num_realizations"]
        parameter_grids = context["model_parameter_grids"]

        simulated_blocks: List[np.ndarray] = []
        labels_list: List[str] = []

        print(f"[TASK] Instantiating factory simulators and calculating paths...")
        for model_cfg in config.models:
            if not model_cfg.enabled:
                print(f" -> Skipping disabled profile tracking: [{model_cfg.name}]")
                continue

            print(f" -> Simulating process tracks for: {model_cfg.name}")

            parameters = parameter_grids[model_cfg.name]
            model_instance = SimulatorFactory.create(process_name=model_cfg.process_enum, parameters=parameters)
            data_matrix = model_instance.simulate(num_steps=context["num_steps"], burn_in=context["burn_in"])

            simulated_blocks.append(data_matrix)
            labels_list.extend([model_cfg.name] * M)

        if not simulated_blocks:
            raise ValueError("Pipeline configuration contains zero active models to simulate.")

        context["data_matrix"] = np.concatenate(simulated_blocks, axis=1)
        context["model_labels"] = np.array(labels_list)
        return context

