"""Stacked 2-Layer GRU Architecture."""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """Stacked 2-Layer GRU with separate regression and classification heads.

    Inputs:
        x: Tensor of shape (batch_size, sequence_length=60, input_size=K)

    Outputs:
        if task == "regression": scalar prediction tensor of shape (batch_size, 1)
        if task == "classification": raw logit tensor of shape (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = "regression",
    ):
        super().__init__()
        if task not in ("regression", "classification"):
            raise ValueError(f"Invalid task '{task}'. Must be 'regression' or 'classification'.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task = task

        # Stacked GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout_layer = nn.Dropout(p=dropout)

        # Output head
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 60, K)
        gru_out, h_n = self.gru(x)
        # Use final layer's hidden state
        out = h_n[-1]  # (batch, hidden_size)
        out = self.dropout_layer(out)
        logits = self.fc(out)  # (batch, 1)
        return logits
