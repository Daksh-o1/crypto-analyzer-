import pytest
import json
import os
import hashlib
from pathlib import Path

def get_project_root():
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def summary_data():
    root = get_project_root()
    summary_path = root / 'experiments' / 'results' / 'ablation_summary.json'
    assert summary_path.exists(), "ablation_summary.json missing"
    with open(summary_path, 'r') as f:
        return json.load(f)

def test_experiments_represented(summary_data):
    experiments = ['EXP_A_PRICE', 'EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL']
    for task in ['regression', 'classification']:
        assert task in summary_data
        for exp in experiments:
            assert exp in summary_data[task]

def test_models_represented(summary_data):
    reg_models = ['naive', 'ridge', 'random_forest', 'xgboost', 'LSTM', 'GRU', '1D-CNN']
    clf_models = ['naive', 'logistic', 'random_forest', 'xgboost', 'LSTM', 'GRU', '1D-CNN']
    experiments = ['EXP_A_PRICE', 'EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL']
    
    for exp in experiments:
        for model in reg_models:
            assert model in summary_data['regression'][exp], f"Model {model} missing from {exp} regression"
            
        for model in clf_models:
            assert model in summary_data['classification'][exp], f"Model {model} missing from {exp} classification"

def test_result_schemas_and_metrics_finite(summary_data):
    experiments = ['EXP_A_PRICE', 'EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL']
    reg_models = ['naive', 'ridge', 'random_forest', 'xgboost', 'LSTM', 'GRU', '1D-CNN']
    clf_models = ['naive', 'logistic', 'random_forest', 'xgboost', 'LSTM', 'GRU', '1D-CNN']
    
    for exp in experiments:
        for model in reg_models:
            metrics = summary_data['regression'][exp][model]
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert isinstance(metrics['mae'], float)
            assert isinstance(metrics['rmse'], float)
            
        for model in clf_models:
            metrics = summary_data['classification'][exp][model]
            assert 'accuracy' in metrics
            assert 'f1_macro' in metrics
            assert 'f1_binary' in metrics
            assert isinstance(metrics['accuracy'], float)
            assert isinstance(metrics['f1_macro'], float)
            if metrics['f1_binary'] is not None:
                assert isinstance(metrics['f1_binary'], float)
                assert metrics['f1_binary'] != metrics['f1_macro'], "f1_binary and f1_macro should not be confused/identical unless mathematically equal by coincidence"

def test_raw_data_integrity():
    root = get_project_root()
    raw_path = root / 'data' / 'raw' / 'btcusdt_5m_raw.csv'
    
    sha256_hash = hashlib.sha256()
    with open(raw_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    assert sha256_hash.hexdigest() == "a722751b2929acdcae48cb875be4c3d4904d17e99d45ea0b3f6d9a67a4cf1eb8", "Raw data mutated!"
