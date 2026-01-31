import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from projects.utils.app_init import build_param_grid
from projects.utils.weights import build_param_str
from projects.utils.eval.dist import (
    compute_confidence,
    compute_margin,
    compute_true_class_output,
    plot_histogram,
)
from projects.utils.eval.filter import apply_filters
from projects.utils.eval.judgment import load_judgment_results

"""
python -m projects.apps.eval_dist_node --config projects/configs/eval_dist_node.yaml
"""


def collect_node_values_for_param(
    param_name: str,
    mode: str,
    judge_dir: Path,
    pred_result_dir: Path,
    filters: dict
) -> list:
    """
    1つのパラメータ組み合わせについて、フィルタリング後のサンプル情報を収集
    指標は収集時に計算し、生データ（predictions, labels）は保持しない
    """
    judgment_csv_path = judge_dir / param_name / f"judgment_results_{mode}.csv"
    if not judgment_csv_path.exists():
        return []
    
    judgment_results = load_judgment_results(judgment_csv_path)
    filtered_results = apply_filters(judgment_results, filters)
    if not filtered_results:
        return []
    
    grouped = defaultdict(lambda: defaultdict(list))
    for r in filtered_results:
        grouped[r["group"]][r["fold_index"]].append({
            "id": r["id"],
            "true_label": r["true_label"],
            "is_correct": r["is_correct"]
        })
    
    sample_data = []
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
                if sample is None:
                    continue
                
                predictions = sample["predictions"]
                labels = sample["labels"]
                
                sample_data.append({
                    "confidence": compute_confidence(predictions),
                    "margin": compute_margin(predictions),
                    "true_class_output": compute_true_class_output(predictions, labels),
                    "true_label": item["true_label"],
                    "is_correct": item["is_correct"]
                })
    
    return sample_data


def aggregate_metrics_by_category(sample_data: list, class_order: list) -> dict:
    """
    サンプルデータを1回のループでカテゴリ別に集約
    """
    categories = {
        "all": {"confidence": [], "margin": [], "true_class_output": []},
        "correct": {"confidence": [], "margin": [], "true_class_output": []},
        "incorrect": {"confidence": [], "margin": [], "true_class_output": []},
    }
    for label_idx in class_order:
        categories[f"label_{label_idx}"] = {"confidence": [], "margin": [], "true_class_output": []}
    
    for sample in sample_data:
        conf = sample["confidence"]
        marg = sample["margin"]
        tco = sample["true_class_output"]
        true_label = sample["true_label"]
        is_correct = sample["is_correct"]
        
        categories["all"]["confidence"].append(conf)
        categories["all"]["margin"].append(marg)
        categories["all"]["true_class_output"].append(tco)
        
        key = "correct" if is_correct else "incorrect"
        categories[key]["confidence"].append(conf)
        categories[key]["margin"].append(marg)
        categories[key]["true_class_output"].append(tco)
        
        categories[f"label_{true_label}"]["confidence"].append(conf)
        categories[f"label_{true_label}"]["margin"].append(marg)
        categories[f"label_{true_label}"]["true_class_output"].append(tco)
    
    result = {}
    for cat_name, metrics in categories.items():
        if metrics["confidence"]:
            result[cat_name] = {
                "confidence": np.concatenate(metrics["confidence"]),
                "margin": np.concatenate(metrics["margin"]),
                "true_class_output": np.concatenate(metrics["true_class_output"]),
            }
        else:
            result[cat_name] = None
    
    return result


def plot_category_metrics(aggregated: dict, cat_key: str, suffix: str, color: str, output_dir: Path, cfg) -> None:
    """
    1つのカテゴリの全指標をプロット
    """
    if aggregated[cat_key] is None:
        print(f"  No data for {suffix}")
        return
    
    metrics_data = aggregated[cat_key]
    n_frames = len(metrics_data["confidence"])
    print(f"  Plotting {suffix} (n_frames={n_frames})...")
    
    metric_configs = {
        "confidence": cfg.metrics["confidence"],
        "margin": cfg.metrics["margin"],
        "true_class_output": cfg.metrics["true_class_output"]
    }
    
    for metric_name, metric_cfg in metric_configs.items():
        if not metric_cfg["enabled"]:
            continue
        
        values = metrics_data[metric_name]
        if values is None or len(values) == 0:
            continue
        
        value_range = tuple(metric_cfg["range"]) if metric_cfg["range"] else (0, 1)
        output_path = output_dir / f"node_{metric_name}_{suffix}.png"
        
        plot_histogram(
            values=values,
            output_path=output_path,
            bins=cfg.bins,
            color=color,
            xlabel=metric_cfg["xlabel"],
            value_range=value_range
        )
        print(f"    Saved: {output_path}")


def process_mode(mode: str, cfg, judge_dir: Path, pred_result_dir: Path, param_grid: list) -> None:
    """
    1つのmodeに対する処理
    """
    print(f"Processing mode: {mode}")
    
    mode_output_dir = cfg.output_dir / mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)
    
    # サンプルデータを収集（指標計算済み）
    all_sample_data = []
    for params in param_grid:
        param_name = build_param_str(params)
        sample_data = collect_node_values_for_param(
            param_name, mode, judge_dir, pred_result_dir, cfg.filters
        )
        all_sample_data.extend(sample_data)
    
    if not all_sample_data:
        print(f"  No data found for mode: {mode}")
        return
    
    print(f"  Total samples: {len(all_sample_data)}")
    
    # 1回の集約で全カテゴリを分類
    print("  Aggregating metrics by category...")
    aggregated = aggregate_metrics_by_category(all_sample_data, cfg.class_order)
    del all_sample_data
    
    # プロット対象の定義: (cat_key, suffix, color)
    plot_targets = [
        ("all", "all", cfg.colors["all"]),
        ("correct", "correct", cfg.colors["correct"]),
        ("incorrect", "incorrect", cfg.colors["incorrect"]),
    ]
    for i, class_name in enumerate(cfg.class_names):
        plot_targets.append((f"label_{cfg.class_order[i]}", class_name, cfg.colors[class_name]))
    
    # 統一ループでプロット
    for cat_key, suffix, color in plot_targets:
        plot_category_metrics(aggregated, cat_key, suffix, color, mode_output_dir, cfg)


def main(cfg):
    judge_dir = Path(cfg.judge_dir)
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)
    
    for mode in cfg.mode:
        process_mode(mode, cfg, judge_dir, pred_result_dir, param_grid)
    
    print("plot finished")

