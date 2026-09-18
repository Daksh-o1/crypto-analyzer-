"""CLI Script to Execute Deep Learning Experiment Matrix (Phase 6)."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from crypto_analyzer.models.dl.config import DLConfig
from crypto_analyzer.models.dl.lstm import LSTMModel
from crypto_analyzer.models.dl.gru import GRUModel
from crypto_analyzer.models.dl.cnn import CNN1DModel
from crypto_analyzer.models.dl.trainer import DLTrainer, set_seed
from crypto_analyzer.models.dl.metrics import compute_dl_metrics

TENSOR_DIR = Path("D:/CryptoAnalyzer/data/processed/tensors")
CHECKPOINT_DIR = Path("D:/CryptoAnalyzer/models/checkpoints")
RESULTS_DIR = Path("D:/CryptoAnalyzer/experiments/results")
STATUS_FILE = RESULTS_DIR / "dl_matrix_status.json"

EXPERIMENTS = [
    "EXP_A_PRICE",
    "EXP_B_PRICE_VOL",
    "EXP_C_TECH_IND",
    "EXP_D_FULL",
]

ARCHITECTURES = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "1D-CNN": CNN1DModel,
}

TASKS = ["regression", "classification"]


def load_tensors(exp_name: str) -> Dict[str, np.ndarray]:
    """Load NPZ tensor archive for a given experiment configuration."""
    tensor_file = TENSOR_DIR / f"{exp_name}_tensors.npz"
    if not tensor_file.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_file}")
    data = np.load(tensor_file)
    return {
        "X_train": data["X_train"],
        "Y_train": data["Y_train"],
        "X_val": data["X_val"],
        "Y_val": data["Y_val"],
        "X_test": data["X_test"],
        "Y_test": data["Y_test"],
    }


def create_dataloaders(
    tensors: Dict[str, np.ndarray],
    task: str,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convert numpy tensors into PyTorch DataLoaders."""
    col_idx = 0 if task == "regression" else 1

    x_tr = torch.tensor(tensors["X_train"], dtype=torch.float32)
    y_tr = torch.tensor(tensors["Y_train"][:, col_idx], dtype=torch.float32)

    x_val = torch.tensor(tensors["X_val"], dtype=torch.float32)
    y_val = torch.tensor(tensors["Y_val"][:, col_idx], dtype=torch.float32)

    x_te = torch.tensor(tensors["X_test"], dtype=torch.float32)
    y_te = torch.tensor(tensors["Y_test"][:, col_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_status_tracker() -> Dict[str, Any]:
    """Load or initialize status tracking manifest."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_status_tracker(status_data: Dict[str, Any]) -> None:
    """Save status tracking manifest atomically."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)


def run_single_experiment(
    arch_name: str,
    exp_name: str,
    task: str,
    tensors: Dict[str, np.ndarray],
    config: DLConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Train and evaluate a single architecture, experiment, and task combination."""
    set_seed(config.seed)

    input_size = tensors["X_train"].shape[2]
    model_cls = ARCHITECTURES[arch_name]

    if arch_name == "1D-CNN":
        model = model_cls(input_size=input_size, channels=config.cnn_channels, dropout=config.dropout, task=task)
    else:
        model = model_cls(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            task=task,
        )

    train_loader, val_loader, test_loader = create_dataloaders(tensors, task, batch_size=config.batch_size)

    arch_tag = arch_name.lower().replace('-', '')
    checkpoint_name = f"{arch_tag}_{exp_name}_{task}.pt"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    trainer = DLTrainer(model=model, task=task, config=config, device=device, arch_name=arch_name, exp_name=exp_name)

    start_time = time.time()
    fit_summary = trainer.fit(train_loader, val_loader, checkpoint_path=checkpoint_path)
    train_time_sec = time.time() - start_time

    val_loss, val_true, val_pred = trainer.evaluate(val_loader)
    test_loss, test_true, test_pred = trainer.evaluate(test_loader)

    val_metrics = compute_dl_metrics(val_true, val_pred, task)
    test_metrics = compute_dl_metrics(test_true, test_pred, task)

    result = {
        "architecture": arch_name,
        "experiment": exp_name,
        "task": task,
        "input_size": input_size,
        "train_time_seconds": round(train_time_sec, 2),
        "best_epoch": fit_summary["best_epoch"],
        "best_val_loss": round(fit_summary["best_val_loss"], 6),
        "checkpoint_path": str(checkpoint_path),
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "history": {
            "train_loss": [round(x, 6) for x in fit_summary["train_history"]],
            "val_loss": [round(x, 6) for x in fit_summary["val_history"]],
        },
    }

    # Explicit cleanup to release RAM
    del model, trainer, train_loader, val_loader, test_loader
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Phase 6 Deep Learning Experiments")
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test run")
    parser.add_argument("--arch", type=str, choices=list(ARCHITECTURES.keys()), help="Run specific architecture only")
    parser.add_argument("--exp", type=str, choices=EXPERIMENTS, help="Run specific experiment only")
    parser.add_argument("--task", type=str, choices=TASKS, help="Run specific task only")
    parser.add_argument("--force", action="store_true", help="Force rerun even if completed")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Controlled threading for CPU stability
    torch.set_num_threads(1)
    device = torch.device("cpu")

    config = DLConfig()
    if args.smoke:
        config.max_epochs = 3
        config.patience = 2
        config.batch_size = 32

    status_tracker = get_status_tracker()

    arch_list = [args.arch] if args.arch else list(ARCHITECTURES.keys())
    exp_list = [args.exp] if args.exp else EXPERIMENTS
    task_list = [args.task] if args.task else TASKS

    total_runs = len(arch_list) * len(exp_list) * len(task_list)
    print(f"=== Starting Phase 6 Deep Learning Matrix Execution (Total Targets: {total_runs}) ===")

    for exp_name in exp_list:
        print(f"\n[DATA] Loading tensors for {exp_name}...")
        tensors = load_tensors(exp_name)

        exp_result_file = RESULTS_DIR / f"{exp_name}_dl_results.json"
        if exp_result_file.exists():
            try:
                with open(exp_result_file, "r") as f:
                    exp_data = json.load(f)
            except Exception:
                exp_data = {}
        else:
            exp_data = {}

        for arch_name in arch_list:
            if arch_name not in exp_data:
                exp_data[arch_name] = {}

            for task in task_list:
                run_key = f"{arch_name.lower().replace('-', '')}_{exp_name}_{task}"

                # Resumability Check
                if not args.force and run_key in status_tracker and status_tracker[run_key].get("status") == "COMPLETED":
                    ckpt_file = Path(status_tracker[run_key].get("checkpoint_path", ""))
                    if ckpt_file.exists():
                        print(f"[SKIP] Run '{run_key}' already COMPLETED. Skipping.")
                        continue

                print(f"\n[RUN START] Architecture={arch_name} | Experiment={exp_name} | Task={task}")
                print(f"[DATA] Train={tensors['X_train'].shape[0]} | Val={tensors['X_val'].shape[0]} | Test={tensors['X_test'].shape[0]} | Input Shape={tensors['X_train'].shape[1:]}")

                status_tracker[run_key] = {"status": "RUNNING", "start_time": time.time()}
                save_status_tracker(status_tracker)

                try:
                    run_res = run_single_experiment(arch_name, exp_name, task, tensors, config, device)
                    exp_data[arch_name][task] = run_res

                    status_tracker[run_key] = {
                        "status": "COMPLETED",
                        "checkpoint_path": run_res["checkpoint_path"],
                        "best_val_loss": run_res["best_val_loss"],
                        "best_epoch": run_res["best_epoch"],
                        "train_time_seconds": run_res["train_time_seconds"],
                    }
                    save_status_tracker(status_tracker)

                    with open(exp_result_file, "w") as f:
                        json.dump(exp_data, f, indent=2)

                    if task == "regression":
                        print(f"[RUN COMPLETE] Best Epoch={run_res['best_epoch']} | Val MAE={run_res['metrics']['validation']['mae']} | Test MAE={run_res['metrics']['test']['mae']}")
                    else:
                        print(f"[RUN COMPLETE] Best Epoch={run_res['best_epoch']} | Val Acc={run_res['metrics']['validation']['accuracy']} | Test Acc={run_res['metrics']['test']['accuracy']}")

                except Exception as e:
                    import traceback
                    err_str = traceback.format_exc()
                    print(f"[RUN FAILED] Run '{run_key}' encountered an exception:\n{err_str}")
                    status_tracker[run_key] = {
                        "status": "FAILED",
                        "error": str(e),
                        "traceback": err_str,
                    }
                    save_status_tracker(status_tracker)

        del tensors
        gc.collect()

    # Print summary counts
    completed = sum(1 for v in status_tracker.values() if v.get("status") == "COMPLETED")
    failed = sum(1 for v in status_tracker.values() if v.get("status") == "FAILED")
    pending = 24 - completed - failed
    print("\n==========================================")
    print(f"MATRIX EXECUTION SUMMARY: COMPLETED: {completed}/24 | FAILED: {failed}/24 | PENDING: {pending}/24")
    print("==========================================")


if __name__ == "__main__":
    main()
