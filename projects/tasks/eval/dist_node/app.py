from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.utils.app_init import build_param_grid
from projects.utils.eval.dist import (
    compute_confidence,
    compute_margin,
    compute_true_class_output,
    plot_confusion_distribution,
    plot_histogram,
)
from projects.utils.eval.input import load_filtered_prediction_samples
from projects.utils.weights import build_param_str

"""
python -m projects.apps.eval_dist_node --config projects/configs/eval_dist_node.yaml
"""


def collect_node_values_for_param(
    param_name: str, mode: str, judge_dir: Path, pred_result_dir: Path, filters: dict
) -> list:
    """
    1つのパラメータ組み合わせについて、フィルタリング後のサンプル情報を収集
    指標は収集時に計算し、生データ（predictions, labels）は保持しない
    """
    sample_data = []
    samples = load_filtered_prediction_samples(
        param_name, mode, judge_dir, pred_result_dir, filters
    )
    for sample in samples:
        predictions = sample["predictions"]
        labels = sample["labels"]

        preds_arr = np.array(predictions)
        n_nodes = preds_arr.shape[1]
        sample_data.append(
            {
                "confidence": compute_confidence(predictions),
                "margin": compute_margin(predictions),
                "true_class_output": compute_true_class_output(predictions, labels),
                "node_outputs": {j: preds_arr[:, j] for j in range(n_nodes)},
                "true_label": sample["true_label"],
                "is_correct": sample["is_correct"],
            }
        )

    return sample_data


def aggregate_metrics_by_category(sample_data: list, class_order: list) -> tuple:
    """
    サンプルデータを1回のループでカテゴリ別に集約。
    Returns:
        result: カテゴリ別指標 {cat_key: {metric: np.ndarray}}
        node_data: 2D集約 {true_label: {node_idx: [values]}}
    """
    categories = {
        "all": {"confidence": [], "margin": [], "true_class_output": []},
        "correct": {"confidence": [], "margin": [], "true_class_output": []},
        "incorrect": {"confidence": [], "margin": [], "true_class_output": []},
    }
    for label_idx in class_order:
        categories[f"label_{label_idx}"] = {"confidence": [], "margin": [], "true_class_output": []}

    node_data = defaultdict(lambda: defaultdict(list))

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

        for node_idx, values in sample["node_outputs"].items():
            node_data[true_label][node_idx].extend(values.tolist())

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

    return result, node_data


def plot_category_metrics(
    aggregated: dict, cat_key: str, suffix: str, color: str, output_dir: Path, cfg
) -> None:
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
        "true_class_output": cfg.metrics["true_class_output"],
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
            value_range=value_range,
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
    aggregated, node_data = aggregate_metrics_by_category(all_sample_data, cfg.class_order)
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

    # 3×3 ノード出力分布の一望プロット
    node_output_cfg = cfg.metrics.get("node_output", {})
    if node_output_cfg.get("enabled", False):
        value_range = tuple(node_output_cfg["range"]) if node_output_cfg.get("range") else (0, 1)
        overview_path = mode_output_dir / f"node_output_confusion_{mode}.png"
        plot_confusion_distribution(
            data=node_data,
            class_names=cfg.class_names,
            class_order=cfg.class_order,
            output_path=overview_path,
            bins=cfg.bins,
            colors=cfg.colors,
            show_count=getattr(cfg, "show_count", True),
            show_cumulative=getattr(cfg, "show_cumulative", False),
            value_range=value_range,
            xlabel=node_output_cfg.get("xlabel", "Node Output Value"),
            col_label="node",
        )
        print(f"  Saved overview: {overview_path}")


def main(cfg):
    judge_dir = Path(cfg.judge_dir)
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)

    for mode in cfg.mode:
        process_mode(mode, cfg, judge_dir, pred_result_dir, param_grid)

    print("plot finished")
