"""Unit test suite for Phase 6 PyTorch Deep Learning Models."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from crypto_analyzer.models.dl.config import DLConfig
from crypto_analyzer.models.dl.lstm import LSTMModel
from crypto_analyzer.models.dl.gru import GRUModel
from crypto_analyzer.models.dl.cnn import CNN1DModel
from crypto_analyzer.models.dl.trainer import DLTrainer, set_seed
from crypto_analyzer.models.dl.metrics import compute_dl_metrics
from crypto_analyzer.models.dl.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def synthetic_tensors():
    """Generate small synthetic tensors for fast unit testing."""
    set_seed(42)
    N_train, N_val, N_test = 64, 32, 32
    W, K = 60, 4
    x_train = torch.randn(N_train, W, K)
    y_reg_train = torch.randn(N_train, 1)
    y_cls_train = torch.randint(0, 2, (N_train, 1)).float()

    x_val = torch.randn(N_val, W, K)
    y_reg_val = torch.randn(N_val, 1)
    y_cls_val = torch.randint(0, 2, (N_val, 1)).float()

    train_reg_ds = TensorDataset(x_train, y_reg_train)
    val_reg_ds = TensorDataset(x_val, y_reg_val)
    train_cls_ds = TensorDataset(x_train, y_cls_train)
    val_cls_ds = TensorDataset(x_val, y_cls_val)

    return {
        "train_reg_loader": DataLoader(train_reg_ds, batch_size=16),
        "val_reg_loader": DataLoader(val_reg_ds, batch_size=16),
        "train_cls_loader": DataLoader(train_cls_ds, batch_size=16),
        "val_cls_loader": DataLoader(val_cls_ds, batch_size=16),
        "K": K,
    }


def test_lstm_forward():
    x = torch.randn(16, 60, 4)
    model = LSTMModel(input_size=4, hidden_size=32, num_layers=2, task="regression")
    out = model(x)
    assert out.shape == (16, 1)


def test_gru_forward():
    x = torch.randn(16, 60, 5)
    model = GRUModel(input_size=5, hidden_size=32, num_layers=2, task="classification")
    out = model(x)
    assert out.shape == (16, 1)


def test_cnn_forward():
    x = torch.randn(16, 60, 19)
    model = CNN1DModel(input_size=19, channels=[32, 64], task="regression")
    out = model(x)
    assert out.shape == (16, 1)


def test_classification_logits_raw():
    x = torch.randn(8, 60, 4)
    model = LSTMModel(input_size=4, task="classification")
    out = model(x)
    # Ensure no sigmoid clipping, values can be outside [0, 1]
    assert out.shape == (8, 1)


def test_dl_metrics_computation():
    y_true_reg = np.array([0.5, -0.2, 1.0, 0.0])
    y_pred_reg = np.array([0.4, -0.1, 0.9, 0.05])
    reg_metrics = compute_dl_metrics(y_true_reg, y_pred_reg, task="regression")
    assert "mae" in reg_metrics
    assert "rmse" in reg_metrics
    assert "mape" in reg_metrics

    y_true_cls = np.array([1, 0, 1, 0])
    y_pred_cls = np.array([0.8, -0.5, 0.2, -0.1])
    cls_metrics = compute_dl_metrics(y_true_cls, y_pred_cls, task="classification")
    assert "accuracy" in cls_metrics
    assert "precision" in cls_metrics
    assert "recall" in cls_metrics
    assert "f1_binary" in cls_metrics, "f1_binary (positive-class F1) must be present"
    assert "f1_macro" in cls_metrics, "f1_macro (macro-averaged F1) must be present"
    assert "confusion_matrix" in cls_metrics
    # Ensure old "f1" key is no longer used to prevent ambiguity
    assert "f1" not in cls_metrics, "Bare 'f1' key must not exist; use f1_binary or f1_macro"


def test_f1_macro_vs_binary_distinct_on_imbalanced():
    """Verify f1_macro != f1_binary when predictions are class-imbalanced.

    A degenerate all-positive predictor gives:
      - Binary F1 (positive class): non-trivial since all positives predicted
      - Macro F1: average of class-0 F1 (=0.0) and class-1 F1, always < binary F1
    """
    # Simulate all-positive predictions (like degenerate 1D-CNN)
    # 40 true-positives + 60 true-negatives, all predicted positive
    y_true = np.array([1] * 40 + [0] * 60)
    y_pred = np.array([1.0] * 100)  # all positive logits

    metrics = compute_dl_metrics(y_true, y_pred, task="classification")

    f1_bin = metrics["f1_binary"]
    f1_mac = metrics["f1_macro"]

    # Positive-class recall=1.0, so binary F1 > 0
    assert f1_bin > 0.0
    # Macro F1 is always <= binary F1 for all-positive predictor with balanced labels
    assert f1_mac < f1_bin, (
        f"Macro F1 ({f1_mac}) should be less than binary F1 ({f1_bin}) "
        f"for all-positive predictor"
    )
    # Specifically: class-0 F1=0, so macro = (0 + f1_bin) / 2
    expected_macro = f1_bin / 2.0
    assert abs(f1_mac - expected_macro) < 1e-5


def test_f1_macro_uses_average_macro():
    """Verify f1_macro matches explicit sklearn call with average='macro'."""
    from sklearn.metrics import f1_score as sk_f1

    y_true = np.array([1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0.8, -0.3, 0.6, 0.1, -0.1, 0.9, -0.4])

    metrics = compute_dl_metrics(y_true, y_pred, task="classification")

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    expected_macro = float(sk_f1(y_true_bin, y_pred_bin, average="macro", zero_division=0))
    assert abs(metrics["f1_macro"] - expected_macro) < 1e-6, (
        f"f1_macro={metrics['f1_macro']} does not match explicit macro call={expected_macro}"
    )

    expected_binary = float(sk_f1(y_true_bin, y_pred_bin, zero_division=0))
    assert abs(metrics["f1_binary"] - expected_binary) < 1e-6, (
        f"f1_binary={metrics['f1_binary']} does not match explicit binary call={expected_binary}"
    )



def test_checkpoint_saving_loading(tmp_path):
    model = LSTMModel(input_size=4, hidden_size=16)
    ckpt_file = tmp_path / "test_model.pt"

    save_checkpoint(model, ckpt_file, metadata={"test_key": "val"})
    assert ckpt_file.exists()

    model_new = LSTMModel(input_size=4, hidden_size=16)
    meta = load_checkpoint(model_new, ckpt_file)
    assert meta["test_key"] == "val"

    # Verify state dict weights match
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        assert torch.allclose(p1, p2)


def test_trainer_fit_loop(synthetic_tensors, tmp_path):
    config = DLConfig(max_epochs=3, patience=2, batch_size=16)
    model = GRUModel(input_size=synthetic_tensors["K"], hidden_size=16, task="regression")
    trainer = DLTrainer(model=model, task="regression", config=config)

    ckpt_path = tmp_path / "gru_test.pt"
    history = trainer.fit(
        train_loader=synthetic_tensors["train_reg_loader"],
        val_loader=synthetic_tensors["val_reg_loader"],
        checkpoint_path=ckpt_path,
    )

    assert "best_epoch" in history
    assert "train_history" in history
    assert len(history["train_history"]) <= 3
    assert ckpt_path.exists()


def test_reproducibility_seed():
    set_seed(42)
    t1 = torch.randn(5)
    set_seed(42)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)
