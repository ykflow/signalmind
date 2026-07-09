import torch.nn as nn
import torch.optim as optim
from utilities.config_reader import MlpClassifierConfig


class OptimizerFactory:
    """Creates a bound PyTorch optimizer using a type-safe MlpClassifierConfig dataclass."""
    def __init__(self, mlp_cfg: MlpClassifierConfig):
        self.train_config = mlp_cfg.training
        self.opt_type = self.train_config.optimizer.lower()
        self.lr = self.train_config.learning_rate
        self.weight_decay = self.train_config.weight_decay
        self.momentum = self.train_config.momentum

    def create(self, model: nn.Module) -> optim.Optimizer:
        if self.opt_type == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_type == 'adam':
            return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_type == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_type == 'sgd':
            return optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        else:
            raise ValueError(f"Unsupported Optimizer designated in configuration: '{self.opt_type}'")