from typing import Any, Dict
from workflows.abstract_workflow import AbstractTask
from utilities.config_reader import ConfigReader



class ReadConfigTask(AbstractTask):
    """TASK: Parses and validates the configuration profile layout file."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[TASK] Loading pipeline configuration from: {self.config_path}")
        config = ConfigReader.load_config(self.config_path)

        context["config"] = config
        context["num_realizations"] = config.num_realizations
        context["num_steps"] = config.num_steps
        context["burn_in"] = config.burn_in
        context["feature_names"] = config.features
        return context

