from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Any
import pyaml
from pathlib import Path
from simulators.enum_models import ProcessName
from feature_extractors.enum_features import FeatureName


@dataclass(frozen=True)
class TimeSeriesModelConfig:
    """Holds structured layout settings and parameter boundary thresholds for a single process model."""
    name: str
    process_enum: ProcessName
    enabled: bool
    parameter_bounds: Dict[str, Tuple[float, float]]


@dataclass(frozen=True)
class MlpTrainingConfig:
    """Holds hyper-parameters specifically bound to the neural optimizer routines."""
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float


@dataclass(frozen=True)
class MlpClassifierConfig:
    """Holds physical architectural configurations for the multi-layer perceptron framework."""
    hidden_layers: List[int]
    dropout_rates: List[float]
    activation: str
    use_batch_norm: bool
    weight_init: str
    training: MlpTrainingConfig


@dataclass(frozen=True)
class SimulationPipelineConfig:
    """Holds global timeline lengths and execution blocks parsed from runtime configuration profiles."""
    num_steps: int
    num_realizations: int
    burn_in: int
    features: Dict[FeatureName, Dict[str, Any]] = field(default_factory=dict)
    models: List[TimeSeriesModelConfig] = field(default_factory=list)
    mlp: MlpClassifierConfig = field(default=None)  # Appended dynamic neural configuration block


class ConfigReader:
    """Parses consolidated simulation, feature extraction, and neural pipeline configurations using pyaml."""

    @staticmethod
    def load_config(file_path: Union[str, Path]) -> SimulationPipelineConfig:
        """Loads and parses the configurations, mapping names to strict runtime Enums and Nested configs."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration profile layout file missing at target: {path.absolute()}")

        with open(path, 'r') as file:
            raw_data = pyaml.yaml.safe_load(file)

        globals_raw = raw_data.get("global_settings", {})

        # 1. Parse Active Features
        parsed_features = {}
        for f in raw_data.get("features", []):
            if f.get("enabled", True):
                raw_name = f.get("name")
                if not raw_name:
                    continue

                try:
                    feature_enum = FeatureName[raw_name.upper()]
                except KeyError:
                    raise ValueError(f"Feature name identifier '{raw_name}' doesn't match feature enums.")
                feature_params = f.get("parameters", {})
                parsed_features[feature_enum] = feature_params

        # 2. Parse Enabled Simulation Models
        parsed_models = []
        for m in raw_data.get("models", []):
            name_str = m.get("name")
            try:
                process_enum = ProcessName.from_string(name_str)
            except ValueError as e:
                raise ValueError(f"YAML Configuration syntax structure fault: {e}")

            bounds = {
                param_name: tuple(bound_range)
                for param_name, bound_range in m.get("parameter_bounds", {}).items()
            }

            parsed_models.append(
                TimeSeriesModelConfig(
                    name=name_str,
                    process_enum=process_enum,
                    enabled=m.get("enabled", True),
                    parameter_bounds=bounds
                )
            )

        # 3. Parse Nested MLP Settings
        mlp_raw = raw_data.get("mlp")
        parsed_mlp = None
        if mlp_raw:
            train_raw = mlp_raw.get("training", {})
            training_config = MlpTrainingConfig(
                optimizer=train_raw.get("optimizer", "adamw"),
                learning_rate=train_raw.get("learning_rate", 0.001),
                weight_decay=train_raw.get("weight_decay", 0.01),
                momentum=train_raw.get("momentum", 0.9)
            )

            parsed_mlp = MlpClassifierConfig(
                hidden_layers=mlp_raw.get("hidden_layers", [256, 128, 64]),
                dropout_rates=mlp_raw.get("dropout_rates", [0.4, 0.2, 0.1]),
                activation=mlp_raw.get("activation", "leaky_relu"),
                use_batch_norm=mlp_raw.get("use_batch_norm", True),
                weight_init=mlp_raw.get("weight_init", "kaiming_normal"),
                training=training_config
            )

        return SimulationPipelineConfig(
            num_steps=globals_raw.get("num_steps", 500),
            num_realizations=globals_raw.get("num_realizations", 1000),
            burn_in=globals_raw.get("burn_in", 250),
            features=parsed_features,
            models=parsed_models,
            mlp=parsed_mlp
        )
