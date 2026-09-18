"""Checkpoint Persistence Utilities."""

import os
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn


CHECKPOINT_DIR = Path("D:/CryptoAnalyzer/models/checkpoints")


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    metadata: Dict[str, Any] = None,
) -> Path:
    """Save PyTorch model state dict and optional metadata."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load PyTorch model state dict from disk."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("metadata", {})
