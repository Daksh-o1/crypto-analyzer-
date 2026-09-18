import json
import pytest
import math
from pathlib import Path

def test_regime_performance():
    results_path = Path("D:/CryptoAnalyzer/experiments/results/regime_model_performance.json")
    assert results_path.exists(), "regime_model_performance.json is missing"
    
    with open(results_path) as f:
        data = json.load(f)
        
    assert "metadata" in data
    assert "results" in data
    
    results = data["results"]
    
    experiments = {"EXP_A_PRICE", "EXP_B_PRICE_VOL", "EXP_C_TECH_IND", "EXP_D_FULL"}
    models_reg = {"naive", "ridge", "random_forest", "xgboost", "LSTM", "GRU", "1D-CNN"}
    models_cls = {"naive", "logistic", "random_forest", "xgboost", "LSTM", "GRU", "1D-CNN"}
    regimes = {"Bullish", "Bearish", "Sideways", "High Volatility"}
    
    for r in results:
        assert r["experiment"] in experiments
        if r["task"] == "regression":
            assert r["model"] in models_reg
        else:
            assert r["model"] in models_cls
        assert r["regime"] in regimes
        
        if r["sample_count"] == 0:
            assert r["metrics"] is None
        else:
            assert r["metrics"] is not None
            if r["task"] == "regression":
                assert "mae" in r["metrics"]
                assert "rmse" in r["metrics"]
                assert "mape" in r["metrics"]
                assert not any(math.isnan(v) or math.isinf(v) for v in r["metrics"].values())
            else:
                assert "accuracy" in r["metrics"]
                assert "precision" in r["metrics"]
                assert "recall" in r["metrics"]
                assert "f1_macro" in r["metrics"]
                assert "confusion_matrix" in r["metrics"]
                m = r["metrics"]
                assert not any(math.isnan(v) or math.isinf(v) for k, v in m.items() if k != "confusion_matrix")

    # Check 1282 alignment for each combination of model+experiment+task
    counts = {}
    for r in results:
        key = (r["experiment"], r["model"], r["task"])
        counts[key] = counts.get(key, 0) + r["sample_count"]
        
    for key, count in counts.items():
        assert count == 1282, f"Total samples for {key} is {count}, expected 1282"
