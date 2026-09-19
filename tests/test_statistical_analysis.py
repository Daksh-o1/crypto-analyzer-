import json
import math
from pathlib import Path
import pytest

RESULTS_DIR = Path("D:/CryptoAnalyzer/experiments/results")

def test_statistical_analysis_output():
    json_path = RESULTS_DIR / "statistical_significance.json"
    assert json_path.exists(), "statistical_significance.json is missing"
    
    with open(json_path) as f:
        data = json.load(f)
        
    assert "metadata" in data
    assert "results" in data
    
    meta = data["metadata"]
    assert meta["test_sample_size"] == 1282
    
    results = data["results"]
    assert len(results) == 24, "Should have exactly 24 targeted comparisons"
    
    for r in results:
        assert "task" in r
        assert "comparison" in r
        assert "statistic" in r
        assert "p_value" in r
        assert "adjusted_p_value" in r
        assert "significant" in r
        assert "n_total" in r
        assert "n_nonzero" in r
        assert "mean_diff" in r
        
        assert r["n_total"] == 1282
        
        if r["n_nonzero"] > 0:
            assert not math.isnan(r["statistic"])
            assert 0.0 <= r["p_value"] <= 1.0
            assert 0.0 <= r["adjusted_p_value"] <= 1.0
        else:
            assert math.isnan(r["statistic"])
            assert r["p_value"] == 1.0
            
        assert isinstance(r["significant"], bool)
