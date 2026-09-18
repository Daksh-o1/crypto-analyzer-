"""Unified PyTorch Model Trainer."""

import random
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from crypto_analyzer.models.dl.config import DLConfig
from crypto_analyzer.models.dl.metrics import compute_dl_metrics
from crypto_analyzer.models.dl.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DLTrainer:
    """Unified PyTorch Training Engine supporting Early Stopping, Scheduler, and Checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        task: str,
        config: DLConfig = None,
        device: torch.device = None,
        arch_name: str = "Unknown",
        exp_name: str = "Unknown",
    ):
        if task not in ("regression", "classification"):
            raise ValueError(f"Invalid task '{task}'. Must be 'regression' or 'classification'.")

        self.config = config or DLConfig()
        set_seed(self.config.seed)

        self.task = task
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.arch_name = arch_name
        self.exp_name = exp_name

        # Select Loss Function according to research spec
        if self.task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch with gradient updates."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            batch_y = batch_y.to(self.device, dtype=torch.float32).view(-1, 1)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        return running_loss / max(1, total_samples)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run evaluation on validation or test set without gradient updates."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32).view(-1, 1)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                running_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

                all_preds.append(outputs.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())

        avg_loss = running_loss / max(1, total_samples)
        y_pred = np.vstack(all_preds).ravel()
        y_true = np.vstack(all_trues).ravel()

        return avg_loss, y_true, y_pred

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """Run complete model training loop with validation, early stopping, and checkpointing."""
        best_val_loss = float("inf")
        patience_counter = 0
        train_history = []
        val_history = []

        checkpoint_path = Path(checkpoint_path)

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            train_history.append(train_loss)
            val_history.append(val_loss)

            improved = False
            # Checkpoint & Early Stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                improved = True
                save_checkpoint(
                    model=self.model,
                    checkpoint_path=checkpoint_path,
                    metadata={
                        "epoch": epoch,
                        "best_val_loss": float(best_val_loss),
                        "task": self.task,
                    },
                )
            else:
                patience_counter += 1

            log_msg = (
                f"[{self.arch_name} | {self.exp_name} | {self.task}] "
                f"Epoch {epoch}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            if improved:
                log_msg += f" -> [NEW BEST] Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}"
            print(log_msg, flush=True)

            if patience_counter >= self.config.patience:
                print(f"[EARLY STOPPING] Triggered at epoch {epoch} (No improvement for {patience_counter} epochs)", flush=True)
                break

        # Reload best performing validation checkpoint
        load_checkpoint(self.model, checkpoint_path, device=self.device)

        print(f"\n[TRAINING SUMMARY] {self.arch_name} | {self.exp_name} | {self.task}")
        print(f"  - Actual epochs trained: {epoch}")
        print(f"  - Best epoch: {best_epoch}")
        print(f"  - Best Val Loss: {best_val_loss:.6f}")
        print(f"  - Early stopping triggered: {patience_counter >= self.config.patience}")
        print(f"  - Checkpoint path: {checkpoint_path}\n", flush=True)

        return {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "train_history": train_history,
            "val_history": val_history,
            "checkpoint_path": str(checkpoint_path),
        }
