import torch
import torch.nn as nn
from utilities.config_reader import MlpClassifierConfig


class SetLayersMlpClassifier(nn.Module):
    """Dynamically builds PyTorch layers using a type-safe MlpClassifierConfig dataclass."""

    activation_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'gelu': nn.GELU(),
        'elu': nn.ELU()}

    def __init__(self, mlp_cfg: MlpClassifierConfig, input_dim: int, num_classes: int):
        super(SetLayersMlpClassifier, self).__init__()

        hidden_layers = mlp_cfg.hidden_layers
        dropout_rates = mlp_cfg.dropout_rates
        activation_type = mlp_cfg.activation.lower()
        activation_fn = self.activation_map.get(activation_type)
        use_batch_norm = mlp_cfg.use_batch_norm
        weight_init = mlp_cfg.weight_init.lower()

        if len(hidden_layers) != len(dropout_rates):
            raise ValueError(
                f"Configuration Mismatch: 'hidden_layers' length ({len(hidden_layers)}) "
                f"must match 'dropout_rates' length ({len(dropout_rates)})"
            )

        layers = []
        current_dim = input_dim
        for hidden_dim, drop_p in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn)
            if drop_p > 0.0:
                layers.append(nn.Dropout(drop_p))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._initialize_weights(weight_init)

    def _initialize_weights(self, strategy: str):
        # in-place operations to initialize modules
        for m in self.modules():  # PyTorch command that automatically loops through every single component in this NN
                if strategy == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif strategy == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
