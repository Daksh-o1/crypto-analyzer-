import json
import os
from pathlib import Path

def main():
    root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    results_dir = root_dir / 'experiments' / 'results'
    tables_dir = root_dir / 'reports' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = ['EXP_A_PRICE', 'EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL']
    expected_k = {'EXP_A_PRICE': 4, 'EXP_B_PRICE_VOL': 5, 'EXP_C_TECH_IND': 19, 'EXP_D_FULL': 24}
    
    # Load baselines
    with open(results_dir / 'all_baselines_results.json', 'r') as f:
        baselines = json.load(f)
        
    dl_results = {}
    for exp in experiments:
        with open(results_dir / f'{exp}_dl_results.json', 'r') as f:
            dl_results[exp] = json.load(f)
            
    summary = {
        'regression': {},
        'classification': {}
    }
    
    # Populate summary
    for exp in experiments:
        summary['regression'][exp] = {}
        summary['classification'][exp] = {}
        
        # Baselines
        base_exp = baselines.get(exp, {})
        for model in ['naive', 'ridge', 'logistic', 'random_forest', 'xgboost']:
            if model not in base_exp:
                continue
                
            model_data = base_exp[model]
            assert model_data['N_test'] == 1282, f"Expected 1282 test samples, got {model_data['N_test']}"
            assert model_data['K'] == expected_k[exp], f"Expected K={expected_k[exp]}, got {model_data['K']}"
            
            test_metrics = model_data.get('test_metrics', {})
            
            if model != 'logistic':
                # Regression
                reg_metrics = test_metrics.get('regression', {})
                if reg_metrics:
                    summary['regression'][exp][model] = {
                        'mae': reg_metrics.get('mae'),
                        'rmse': reg_metrics.get('rmse'),
                        'mape': reg_metrics.get('mape')
                    }
                    
            if model != 'ridge':
                # Classification
                clf_metrics = test_metrics.get('classification', {})
                if clf_metrics:
                    summary['classification'][exp][model] = {
                        'accuracy': clf_metrics.get('accuracy'),
                        'precision': clf_metrics.get('precision'),
                        'recall': clf_metrics.get('recall'),
                        'f1_binary': clf_metrics.get('f1_binary'),
                        'f1_macro': clf_metrics.get('f1_macro')
                    }
                    if 'f1_binary' not in clf_metrics:
                        summary['classification'][exp][model]['f1_binary'] = None

        # Deep Learning
        dl_exp = dl_results.get(exp, {})
        for model in ['LSTM', 'GRU', '1D-CNN']:
            if model not in dl_exp:
                continue
                
            model_data = dl_exp[model]
            if 'regression' in model_data:
                reg_metrics = model_data['regression']['metrics']['test']
                assert model_data['regression']['input_size'] == expected_k[exp]
                summary['regression'][exp][model] = {
                    'mae': reg_metrics.get('mae'),
                    'rmse': reg_metrics.get('rmse'),
                    'mape': reg_metrics.get('mape')
                }
            if 'classification' in model_data:
                clf_metrics = model_data['classification']['metrics']['test']
                assert model_data['classification']['input_size'] == expected_k[exp]
                summary['classification'][exp][model] = {
                    'accuracy': clf_metrics.get('accuracy'),
                    'precision': clf_metrics.get('precision'),
                    'recall': clf_metrics.get('recall'),
                    'f1_binary': clf_metrics.get('f1_binary'),
                    'f1_macro': clf_metrics.get('f1_macro')
                }

    with open(results_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Tables
    models_reg = ['naive', 'ridge', 'random_forest', 'xgboost', 'LSTM', 'GRU', '1D-CNN']
    models_clf = ['naive', 'logistic', 'random_forest', 'xgboost', 'LSTM', 'GRU', '1D-CNN']

    def format_metric(val):
        return f"{val:.4f}" if val is not None else "N/A"

    def format_change(val, base_val, lower_is_better):
        if val is None or base_val is None:
            return "N/A"
        diff = val - base_val
        return f"{diff:+.4f}"

    # Regression Table
    reg_md = "# Phase 7 Regression Ablation Results\n\n"
    reg_md += "| Model | Metric | EXP_A (Base) | EXP_B | \u0394 EXP_B | EXP_C | \u0394 EXP_C | EXP_D | \u0394 EXP_D |\n"
    reg_md += "|---|---|---|---|---|---|---|---|---|\n"
    for model in models_reg:
        for metric in ['mae', 'rmse']:
            base_val = summary['regression']['EXP_A_PRICE'].get(model, {}).get(metric)
            row = [model if metric == 'mae' else "", metric.upper(), format_metric(base_val)]
            for exp in ['EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL']:
                val = summary['regression'][exp].get(model, {}).get(metric)
                row.append(format_metric(val))
                row.append(format_change(val, base_val, lower_is_better=True))
            reg_md += "| " + " | ".join(row) + " |\n"
            
    with open(tables_dir / 'ablation_regression_comparison.md', 'w', encoding='utf-8') as f:
        f.write(reg_md)
        
    # Classification Table
    clf_md = "# Phase 7 Classification Ablation Results\n\n"
    clf_md += "| Model | Metric | EXP_A (Base) | EXP_B | \u0394 EXP_B | EXP_C | \u0394 EXP_C | EXP_D | \u0394 EXP_D |\n"
    clf_md += "|---|---|---|---|---|---|---|---|---|\n"
    for model in models_clf:
        for metric in ['accuracy', 'f1_macro', 'f1_binary']:
            base_val = summary['classification']['EXP_A_PRICE'].get(model, {}).get(metric)
            row = [model if metric == 'accuracy' else "", metric.upper(), format_metric(base_val)]
            for exp in ['EXP_B_PRICE_VOL', 'EXP_C_TECH_IND', 'EXP_D_FULL']:
                val = summary['classification'][exp].get(model, {}).get(metric)
                row.append(format_metric(val))
                row.append(format_change(val, base_val, lower_is_better=False))
            clf_md += "| " + " | ".join(row) + " |\n"
            
    with open(tables_dir / 'ablation_classification_comparison.md', 'w', encoding='utf-8') as f:
        f.write(clf_md)

if __name__ == "__main__":
    main()
