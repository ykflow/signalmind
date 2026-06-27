from typing import Any, Dict
from workflows.abstract_workflow import AbstractWorkflow
from workflows.workflow_tasks.read_config_task import ReadConfigTask
from workflows.workflow_tasks.generate_parameters_task import GenerateModelParameterTask
from workflows.workflow_tasks.simulate_models_task import SimulateModelsTask
from workflows.workflow_tasks.extract_features_task import ExtractFeaturesTask


class MainSignalMindWorkflow(AbstractWorkflow):
    """ Workflow orchestrator running a main tasks """

    def __init__(self, config_path: str = "config.yaml") -> None:
        super().__init__()
        self.add_task(ReadConfigTask(config_path))
        self.add_task(GenerateModelParameterTask())
        self.add_task(SimulateModelsTask())
        self.add_task(ExtractFeaturesTask())

    def run(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        if initial_context is None:
            initial_context = {}

        context = initial_context
        for task in self._tasks:
            context = task.execute(context)

        return context
