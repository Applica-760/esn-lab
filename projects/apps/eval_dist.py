import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import csv
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from projects.utils.app_init import setup_app_environment, build_param_grid
from projects.utils.weights import build_param_str
from projects.utils.eval.dist import count_true_pred_ratio, plot_histogram
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
    judgment_csv_path = judge_dir / param_name / f"judgment_results_{mode}.csv"
    if not judgment_csv_path.exists():
        return []
    
    judgment_results = load_judgment_results(judgment_csv_path)
    filtered_results = apply_filters(judgment_results, filters)
    if not filtered_results:
        return []
    
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


def one_process(params, mode, judge_dir, pred_result_dir, filters, intermediate_dir):
    param_name = build_param_str(params)
    csv_path = intermediate_dir / f"{param_name}_{mode}_ratios.csv"
    
    if csv_path.exists():
        print(f"skipped: {param_name} {mode}")
        return
    
    ratio_results = collect_ratios_for_param(
        param_name, mode, Path(judge_dir), Path(pred_result_dir), filters
    )
    
    if not ratio_results:
        print(f"no data: {param_name} {mode}")
        return
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ratio", "true_label"])
        writer.writeheader()
        writer.writerows(ratio_results)
    
    print(f"proceed: {param_name} {mode} (n={len(ratio_results)})")


def main():
    cfg, output_dir = setup_app_environment()
    intermediate_dir = output_dir / "intermediate"
    param_grid = build_param_grid(cfg)
    jobs = [(params, mode) for params in param_grid for mode in cfg.mode]
    
    with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
        futures = [
            executor.submit(one_process, params, mode, cfg.judge_dir, cfg.pred_result_dir, cfg.filters, intermediate_dir)
            for params, mode in jobs
        ]
        for future in futures:
            future.result()
    
    for mode in cfg.mode:
        print(f"\nPlotting mode: {mode}")
        all_ratio_results = []
        
        for params in param_grid:
            param_name = build_param_str(params)
            csv_path = intermediate_dir / f"{param_name}_{mode}_ratios.csv"
            if not csv_path.exists():
                continue
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_ratio_results.append({"ratio": float(row["ratio"]), "true_label": int(row["true_label"])})
        
        if not all_ratio_results:
            print(f"  No data found for mode: {mode}")
            continue
        
        all_ratios = [r["ratio"] for r in all_ratio_results]
        output_path = output_dir / f"dist_all_{mode}.png"
        plot_histogram(all_ratios, output_path, cfg.bins, cfg.colors["all"])
        print(f"  Saved: {output_path}")
        
        for i, class_name in enumerate(cfg.class_names):
            label_index = cfg.class_order[i]
            label_ratios = [r["ratio"] for r in all_ratio_results if r["true_label"] == label_index]
            
            if not label_ratios:
                print(f"  No data for {class_name}")
                continue
            
            output_path = output_dir / f"dist_{class_name}_{mode}.png"
            plot_histogram(label_ratios, output_path, cfg.bins, cfg.colors[class_name])
            print(f"  Saved: {output_path} (n={len(label_ratios)})")
    
    print("plot finished")


if __name__ == "__main__":
    main()