import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path

from projects.utils.app_init import setup_app_environment, build_param_grid
from projects.utils.weights import build_param_str
from projects.utils.eval.filter import apply_filters, apply_sampling, extract_ids_with_metadata, group_targets_by_source
from projects.utils.eval.judgment import load_judgment_results
from projects.utils.eval.plot_prediction import plot_prediction


"""
python -m projects.apps.eval_plot --config projects/configs/eval_plot.yaml
"""

def execute_prediction_plots(cfg, param_name, targets):
    """
    プロット実行メインループ
    """
    output_dir = Path(cfg.output_dir)
    grouped = group_targets_by_source(targets)
    
    for group, folds in grouped.items():
        for fold_index, ids in folds.items():
            json_path = Path(cfg.pred_dir) / group / param_name / f"{cfg.mode}_results.json"
            with open(json_path, 'r') as f:
                pred_results = json.load(f)
            
            fold_data = next(fd for fd in pred_results if fd["fold_index"] == fold_index)
            results = fold_data["results"]
            
            for sample_id in ids:
                output_path = output_dir / param_name / group / f"fold_{fold_index}" / sample_id
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plot_prediction(results, sample_id, str(output_path), ext="png")
    

def main():
    # データロード
    cfg, _ = setup_app_environment()
    param_name = build_param_str(build_param_grid(cfg)[0])
    judgment_csv_path = Path(cfg.eval_dir) / param_name / f"judgment_results_{cfg.mode}.csv"
    judgment_results = load_judgment_results(judgment_csv_path)

    # フィルタリング
    filtered_results = apply_filters(judgment_results, cfg.filters)
    sampled_results = apply_sampling(filtered_results, cfg.sampling)
    targets = extract_ids_with_metadata(sampled_results)
    
    # 実行
    execute_prediction_plots(cfg, param_name, targets)
    print("plot finished")

if __name__ == "__main__":
    main()