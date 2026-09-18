"""Centralized Deep Learning Configurations."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DLConfig:
    """Configuration container for PyTorch deep learning models and training."""

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 7
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128])
    seed: int = 42
    weight_decay: float = 1e-4
