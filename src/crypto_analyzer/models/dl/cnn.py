"""Temporal 1D Convolutional Neural Network (1D-CNN)."""

import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    """1D Convolutional Neural Network for sequence forecasting.

    Inputs:
        x: Tensor of shape (batch_size, sequence_length=60, input_size=K)

    Outputs:
        if task == "regression": scalar prediction tensor of shape (batch_size, 1)
        if task == "classification": raw logit tensor of shape (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int,
        channels: list = None,
        dropout: float = 0.2,
        task: str = "regression",
    ):
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"Invalid task '{task}'. Must be 'regression' or 'classification'.")

        if channels is None:
            channels = [64, 128]

        self.input_size = input_size
        self.channels = channels
        self.task = task

        # Conv Block 1
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=channels[0],
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu1 = nn.ReLU()

        # Conv Block 2
        self.conv2 = nn.Conv1d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.relu2 = nn.ReLU()

        # Adaptive Pooling across time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout_layer = nn.Dropout(p=dropout)

        # Output head
        self.fc = nn.Linear(channels[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 60, K) -> permute to (batch, K, 60) for Conv1d
        x = x.permute(0, 2, 1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.pool(out)  # (batch, channels[1], 1)
        out = out.squeeze(-1)  # (batch, channels[1])
        out = self.dropout_layer(out)

        logits = self.fc(out)  # (batch, 1)
        return logits
