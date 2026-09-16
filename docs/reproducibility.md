# Reproducibility Protocol (docs/reproducibility.md)

To ensure scientific reproducibility across environments:

1. **Fixed Random Seeds:** Python (`random`), NumPy (`np.random.seed`), PyTorch (`torch.manual_seed`), and CUDNN determinism flags are explicitly set via configuration files.
2. **Configuration-Driven Experiments:** All experiment parameters, dataset versions, lookback windows, and model hyperparameter settings are defined in YAML config files in `experiments/configs/`.
3. **Artifact Persistence:** Trained model weights (`models/checkpoints/`), prediction arrays, and evaluation JSON metrics (`experiments/results/`) are saved with unique experiment IDs.
4. **Environment Lock:** Dependencies and exact versions are recorded in `pyproject.toml` and `requirements.txt`.
