import json
import csv
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from projects.utils.app_init import build_param_grid
from projects.utils.weights import build_param_str
from projects.utils.eval.dist import count_all_class_ratios, plot_histogram, plot_confusion_distribution
from projects.utils.eval.filter import apply_filters
from projects.utils.eval.judgment import load_judgment_results

"""
python -m projects.apps.eval_dist --config projects/configs/eval_dist.yaml

目的：それぞれどのような割合で，判別に成功，失敗しているのかなどの分析を行う．
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
                ratios, _ = count_all_class_ratios(sample["predictions"], sample["labels"])
                ratio_results.append({"true_label": item["true_label"], "ratios": ratios})
    
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

    n_classes = len(ratio_results[0]["ratios"])
    fieldnames = ["true_label"] + [f"ratio_{j}" for j in range(n_classes)]
    rows = [
        {"true_label": r["true_label"], **{f"ratio_{j}": r["ratios"][j] for j in range(n_classes)}}
        for r in ratio_results
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"proceed: {param_name} {mode} (n={len(ratio_results)})")


def main(cfg):
    intermediate_dir = cfg.output_dir / "intermediate"
    param_grid = build_param_grid(cfg)
    jobs = [(params, mode) for params in param_grid for mode in cfg.mode]
    
    with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
        futures = [
            executor.submit(one_process, params, mode, cfg.judge_dir, cfg.pred_result_dir, cfg.filters, intermediate_dir)
            for params, mode in jobs
        ]
        for future in futures:
            future.result()
    
    n_classes = len(cfg.class_order)

    for mode in cfg.mode:
        print(f"\nPlotting mode: {mode}")

        # CSV を読み込んで data[true_label][pred_class] = [ratios] に集約
        data = defaultdict(lambda: defaultdict(list))
        total = 0
        for params in param_grid:
            param_name = build_param_str(params)
            csv_path = intermediate_dir / f"{param_name}_{mode}_ratios.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    true_label = int(row["true_label"])
                    for j in range(n_classes):
                        data[true_label][j].append(float(row[f"ratio_{j}"]))
                    total += 1

        if total == 0:
            print(f"  No data found for mode: {mode}")
            continue

        print(f"  Total samples: {total}")

        # 個別プロット: N×N = 9枚
        individual_dir = cfg.output_dir / mode / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for row_i, row_name in enumerate(cfg.class_names):
            true_idx = cfg.class_order[row_i]
            for col_j, col_name in enumerate(cfg.class_names):
                pred_idx = cfg.class_order[col_j]
                ratios = data.get(true_idx, {}).get(pred_idx, [])
                if not ratios:
                    print(f"  No data: true={row_name} pred={col_name}")
                    continue
                output_path = individual_dir / f"dist_true{row_name}_argmax{col_name}.png"
                plot_histogram(
                    ratios, output_path, cfg.bins, cfg.colors[col_name],
                    xlabel=f"ratio (pred={col_name})",
                    show_count=cfg.show_count, show_cumulative=cfg.show_cumulative
                )
                print(f"  Saved: {output_path} (n={len(ratios)})")

        # 一望プロット: 3×3 を1枚
        overview_path = cfg.output_dir / mode / f"dist_confusion_{mode}.png"
        plot_confusion_distribution(
            data=data,
            class_names=cfg.class_names,
            class_order=cfg.class_order,
            output_path=overview_path,
            bins=cfg.bins,
            colors=cfg.colors,
            show_count=cfg.show_count,
            show_cumulative=cfg.show_cumulative,
        )
        print(f"  Saved overview: {overview_path}")

    print("plot finished")

