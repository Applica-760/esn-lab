import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path
from collections import defaultdict

from projects.utils.app_init import setup_app_environment, build_param_grid
from projects.utils.weights import build_param_str
from projects.utils.eval.dist import count_true_pred_ratio, plot_ratio_histogram
from projects.utils.eval.filter import apply_filters
from projects.utils.eval.judgment import load_judgment_results

"""
python -m projects.apps.eval_dist --config projects/configs/eval_dist.yaml

目的：それぞれどのような割合で，判別に成功，失敗しているのかなどの分析を行う．
柔軟なパターンがプロットできるように，config側で条件を組み合わせながら色々プロットしたい
これにより，
「ラベルでフィルターをかけてプロットしたら，特定のこのラベルは判別が難しいことがわかった」
「判別に失敗しているものでフィルターをかけたら，意外と多数決にギリギリ負けているだけで，惜しいものが多いことがわかった」
などの考察ができるようにすることを目標とする．
"""


def collect_ratios_for_param(
    param_name: str,
    mode: str,
    judge_dir: Path,
    pred_result_dir: Path,
    filters: dict
) -> list:
    """
    1つのパラメータ組み合わせについて、フィルタリング後のratioリストを収集
    """
    # judgment_resultsをロード
    judgment_csv_path = judge_dir / param_name / f"judgment_results_{mode}.csv"
    if not judgment_csv_path.exists():
        return []
    
    judgment_results = load_judgment_results(judgment_csv_path)
    filtered_results = apply_filters(judgment_results, filters)
    if not filtered_results:
        return []
    
    # group, fold_indexごとにIDをグルーピング
    grouped = defaultdict(lambda: defaultdict(list))
    for r in filtered_results:
        grouped[r["group"]][r["fold_index"]].append({"id": r["id"], "true_label": r["true_label"]})
    
    ratio_results = []
    for group, folds in grouped.items():
        json_path = pred_result_dir / group / param_name / f"{mode}_results.json"
        if not json_path.exists():
            continue
        
        with open(json_path, 'r') as f:
            pred_results = json.load(f)
        
        for fold_index, items in folds.items():
            fold_data = next((fd for fd in pred_results if fd["fold_index"] == fold_index), None)
            if fold_data is None:
                continue
            
            results_by_id = {s["id"]: s for s in fold_data["results"]}
            for item in items:
                sample = results_by_id.get(item["id"])
                
                ratio = count_true_pred_ratio(sample["predictions"], sample["labels"])
                ratio_results.append({"ratio": ratio, "true_label": item["true_label"]})
    
    return ratio_results


def main():
    cfg, output_dir = setup_app_environment()
    
    judge_dir = Path(cfg.judge_dir)
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)
    
    for mode in cfg.mode:
        print(f"Processing mode: {mode}")
        
        # 全パラメータ組み合わせについてデータを合算
        all_ratio_results = []
        for params in param_grid:
            param_name = build_param_str(params)
            ratio_results = collect_ratios_for_param(
                param_name, mode, judge_dir, pred_result_dir, cfg.filters
            )
            all_ratio_results.extend(ratio_results)
        
        if not all_ratio_results:
            print(f"  No data found for mode: {mode}")
            continue
        
        # 全サンプルのヒストグラム
        all_ratios = [r["ratio"] for r in all_ratio_results]
        output_path = output_dir / f"dist_all_{mode}.png"
        plot_ratio_histogram(all_ratios, output_path, cfg.bins, cfg.colors["all"])
        print(f"  Saved: {output_path}")
        
        # true_labelごとのヒストグラム
        for i, class_name in enumerate(cfg.class_names):
            label_index = cfg.class_order[i]  # 表示名に対応するJSONのインデックス
            label_ratios = [r["ratio"] for r in all_ratio_results if r["true_label"] == label_index]
            
            if not label_ratios:
                print(f"  No data for {class_name}")
                continue
            
            output_path = output_dir / f"dist_{class_name}_{mode}.png"
            plot_ratio_histogram(label_ratios, output_path, cfg.bins, cfg.colors[class_name])
            print(f"  Saved: {output_path} (n={len(label_ratios)})")
    
    print("plot finished")


if __name__ == "__main__":
    main()