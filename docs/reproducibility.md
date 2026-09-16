# Reproducibility Protocol (docs/reproducibility.md)

To guarantee exact scientific reproducibility across execution environments:

## 1. Fixed Random Seeds
- **Python Standard Library:** `random.seed(42)`
- **NumPy:** `np.random.seed(42)`
- **PyTorch:** `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`
- **PyTorch Determinism Flags:**
  ```python
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

## 2. Configuration-Driven Pipeline
- All pipeline parameters, lookback window sizes ($W=60$), split ratios ($70/15/15$), regime lookback ($W=288$), and model hyperparameters are defined in structured YAML configuration files inside `experiments/configs/`.
- Zero hardcoded magic numbers inside source code execution paths.

## 3. Train-Only Scaler Persistence
- Feature scalers are fit strictly on training partitions and saved as serialized binary artifacts (`models/checkpoints/scaler_exp_{id}.joblib`).
- Re-running evaluation on test sets reloads the exact training scaler to prevent data distribution leakage.

## 4. Environment & Artifact Logging
- Package dependencies are strictly pinned in `requirements.txt` and `pyproject.toml`.
- Every experiment run logs Git commit SHA, Python version, OS platform, PyTorch version, and execution timestamp into `experiments/runs/`.
