"""Phase 6 Metric Recalculation Script.

DOES NOT RETRAIN. Loads existing checkpoints, runs deterministic forward
passes through the test/val sets, and recomputes classification metrics
using the corrected compute_dl_metrics() that now includes both f1_binary
and f1_macro.

Regression metrics are verified to be unchanged.
Classification metrics gain f1_macro and rename f1 -> f1_binary.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from crypto_analyzer.models.dl.config import DLConfig
from crypto_analyzer.models.dl.lstm import LSTMModel
from crypto_analyzer.models.dl.gru import GRUModel
from crypto_analyzer.models.dl.cnn import CNN1DModel
from crypto_analyzer.models.dl.checkpoint import load_checkpoint
from crypto_analyzer.models.dl.metrics import compute_dl_metrics
from crypto_analyzer.models.dl.trainer import set_seed

TENSOR_DIR = Path("D:/CryptoAnalyzer/data/processed/tensors")
CHECKPOINT_DIR = Path("D:/CryptoAnalyzer/models/checkpoints")
RESULTS_DIR = Path("D:/CryptoAnalyzer/experiments/results")

EXPERIMENTS = ["EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"]
ARCHITECTURES = {"LSTM": LSTMModel, "GRU": GRUModel, "1D-CNN": CNN1DModel}
CONFIG = DLConfig()

TOLERANCE = 1e-4  # acceptable floating-point drift for "unchanged" checks


def load_tensors(exp_name: str) -> dict:
    tensor_file = TENSOR_DIR / f"{exp_name}_tensors.npz"
    data = np.load(tensor_file)
    return {k: data[k] for k in ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test"]}


def make_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=64, shuffle=False)


def evaluate_no_grad(model: torch.nn.Module, loader: DataLoader) -> tuple:
    """Run inference deterministically. Returns (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            out = model(batch_x.to(torch.float32))
            all_preds.append(out.cpu().numpy().ravel())
            all_trues.append(batch_y.cpu().numpy().ravel())
    return (
        np.concatenate(all_trues),
        np.concatenate(all_preds),
    )


def build_model(arch_name: str, input_size: int, task: str) -> torch.nn.Module:
    if arch_name == "1D-CNN":
        return CNN1DModel(input_size=input_size, channels=CONFIG.cnn_channels,
                          dropout=CONFIG.dropout, task=task)
    return ARCHITECTURES[arch_name](
        input_size=input_size,
        hidden_size=CONFIG.hidden_size,
        num_layers=CONFIG.num_layers,
        dropout=CONFIG.dropout,
        task=task,
    )


def assert_close(name: str, old: float, new: float) -> None:
    diff = abs(old - new)
    status = "[OK]" if diff < TOLERANCE else f"[CHANGED by {diff:.8f}]"
    print(f"    {name}: old={old:.6f}  new={new:.6f}  {status}")
    if diff >= TOLERANCE:
        raise RuntimeError(f"Metric '{name}' changed unexpectedly: {old} -> {new}")


def main():
    set_seed(42)
    torch.set_num_threads(1)

    total_cls_processed = 0
    total_reg_verified = 0
    all_macro_f1: dict = {}

    for exp_name in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"[EXPERIMENT] {exp_name}")
        print(f"{'='*60}")

        tensors = load_tensors(exp_name)
        result_file = RESULTS_DIR / f"{exp_name}_dl_results.json"

        with open(result_file, "r") as f:
            exp_data = json.load(f)

        for arch_name in ["LSTM", "GRU", "1D-CNN"]:
            input_size = tensors["X_train"].shape[2]

            # ── REGRESSION ──────────────────────────────────────────────────
            print(f"\n  [{arch_name} | regression] — Verifying regression metrics unchanged")
            arch_tag = arch_name.lower().replace("-", "")
            ckpt_path = CHECKPOINT_DIR / f"{arch_tag}_{exp_name}_regression.pt"

            model_reg = build_model(arch_name, input_size, "regression")
            load_checkpoint(model_reg, ckpt_path, device=torch.device("cpu"))

            val_true, val_pred = evaluate_no_grad(
                model_reg,
                make_loader(tensors["X_val"], tensors["Y_val"][:, 0])
            )
            test_true, test_pred = evaluate_no_grad(
                model_reg,
                make_loader(tensors["X_test"], tensors["Y_test"][:, 0])
            )

            new_val_reg = compute_dl_metrics(val_true, val_pred, "regression")
            new_test_reg = compute_dl_metrics(test_true, test_pred, "regression")

            old_val_reg = exp_data[arch_name]["regression"]["metrics"]["validation"]
            old_test_reg = exp_data[arch_name]["regression"]["metrics"]["test"]

            for key in ("mae", "rmse", "mape"):
                assert_close(f"val.{key}", old_val_reg[key], new_val_reg[key])
                assert_close(f"test.{key}", old_test_reg[key], new_test_reg[key])

            # Regression metrics are confirmed unchanged — no update needed
            total_reg_verified += 1

            # ── CLASSIFICATION ───────────────────────────────────────────────
            print(f"\n  [{arch_name} | classification] — Recomputing classification metrics")
            ckpt_path_cls = CHECKPOINT_DIR / f"{arch_tag}_{exp_name}_classification.pt"

            model_cls = build_model(arch_name, input_size, "classification")
            load_checkpoint(model_cls, ckpt_path_cls, device=torch.device("cpu"))

            val_true_c, val_pred_c = evaluate_no_grad(
                model_cls,
                make_loader(tensors["X_val"], tensors["Y_val"][:, 1])
            )
            test_true_c, test_pred_c = evaluate_no_grad(
                model_cls,
                make_loader(tensors["X_test"], tensors["Y_test"][:, 1])
            )

            new_val_cls = compute_dl_metrics(val_true_c, val_pred_c, "classification")
            new_test_cls = compute_dl_metrics(test_true_c, test_pred_c, "classification")

            old_val_cls = exp_data[arch_name]["classification"]["metrics"]["validation"]
            old_test_cls = exp_data[arch_name]["classification"]["metrics"]["test"]

            # Verify unchanged fields
            for key in ("accuracy", "precision", "recall"):
                assert_close(f"val.{key}", old_val_cls[key], new_val_cls[key])
                assert_close(f"test.{key}", old_test_cls[key], new_test_cls[key])

            # Verify f1_binary == old "f1"
            assert_close("val.f1_binary vs old f1",
                         old_val_cls["f1"] if "f1" in old_val_cls else old_val_cls["f1_binary"],
                         new_val_cls["f1_binary"])
            assert_close("test.f1_binary vs old f1",
                         old_test_cls["f1"] if "f1" in old_test_cls else old_test_cls["f1_binary"],
                         new_test_cls["f1_binary"])

            # Report new macro F1
            key_label = f"{arch_name}_{exp_name}_classification"
            all_macro_f1[key_label] = {
                "val_f1_macro": new_val_cls["f1_macro"],
                "test_f1_macro": new_test_cls["f1_macro"],
            }
            print(f"    NEW val f1_macro  = {new_val_cls['f1_macro']:.6f}")
            print(f"    NEW test f1_macro = {new_test_cls['f1_macro']:.6f}")

            # Write corrected metrics back into exp_data
            exp_data[arch_name]["classification"]["metrics"]["validation"] = new_val_cls
            exp_data[arch_name]["classification"]["metrics"]["test"] = new_test_cls
            total_cls_processed += 1

        # Save updated result file
        with open(result_file, "w") as f:
            json.dump(exp_data, f, indent=2)
        print(f"\n  [SAVED] {result_file}")

    # Summary
    print(f"\n{'='*60}")
    print("RECALCULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Regression results verified unchanged: {total_reg_verified}/12")
    print(f"Classification results recalculated:   {total_cls_processed}/12")
    print("\nAll test f1_macro values:")
    for k, v in all_macro_f1.items():
        print(f"  {k}: {v['test_f1_macro']:.6f}")
    print("\n✅ Recalculation complete. No retraining performed.")


if __name__ == "__main__":
    main()
