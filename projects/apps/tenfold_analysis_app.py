import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import shutil
import numpy as np
from pathlib import Path

from projects.utils.config import load_config
from projects.utils.confusion import (
    compute_cm_from_judgment_results,
    save_confusion_matrix_csv,
    plot_confusion_matrix,
)
from projects.utils.metrics import plot_performance_summary
from projects.utils.results import (
    load_eval_results,
    save_judgment_results,
)
from projects.utils.judgment import compute_judgment_results
from projects.utils.weights import list_param_dirs


"""
python -m projects.apps.tenfold_analysis_app --config projects/configs/tenfold_analysis.yaml
"""


def compute_and_save_judgments(param_dirs, sample_groups, mode, eval_result_dir, output_dir):
    """
    判定結果の計算と保存
    """
    param_judgment_results = {param_dir.name: [] for param_dir in param_dirs}

    for group in sample_groups:
        for param_dir in param_dirs:
            param_name = param_dir.name
            json_path = eval_result_dir / group / param_name / f"{mode}_results.json"

            if not json_path.exists():
                continue

            eval_results = load_eval_results(str(json_path))
            judgment_results = compute_judgment_results(eval_results, group=group)
            param_judgment_results[param_name].extend(judgment_results)

    # 判定結果を保存
    for param_dir in param_dirs:
        param_name = param_dir.name
        judgment_results = param_judgment_results[param_name]
        if judgment_results:
            output_path = output_dir / param_name / "judgment_results"
            save_judgment_results(judgment_results, str(output_path))

    return param_judgment_results


def compute_and_save_confusion_matrices(param_dirs, sample_groups, param_judgment_results, 
                                        output_dir, class_names, class_order):
    """
    第2段階: 混同行列の計算とプロット
    """
    n_classes = len(class_names)
    
    for param_dir in param_dirs:
        param_name = param_dir.name
        judgment_results = param_judgment_results[param_name]
        if not judgment_results:
            continue
            
        output_param_dir = output_dir / param_name
        
        # グループ別の混同行列
        for group in sample_groups:
            group_judgment_results = [r for r in judgment_results if r.get("group") == group]
            if not group_judgment_results:
                continue
            
            # スキップ判定
            accumulated_path = output_param_dir / group / "accumulated.csv"
            if accumulated_path.exists():
                continue
            
            # fold別
            fold_indices = sorted(set(r["fold_index"] for r in group_judgment_results))
            for fold_idx in fold_indices:
                cm = compute_cm_from_judgment_results(group_judgment_results, n_classes, 
                                                     fold_index=fold_idx, group=group)
                output_path = output_param_dir / group / f"fold_{fold_idx}"
                save_confusion_matrix_csv(cm, class_names, str(output_path), class_order)
                plot_confusion_matrix(cm, class_names, f"{param_name} / {group} / fold {fold_idx}", 
                                    str(output_path), class_order)
            
            # accumulated
            cm = compute_cm_from_judgment_results(group_judgment_results, n_classes, group=group)
            output_path = output_param_dir / group / "accumulated"
            save_confusion_matrix_csv(cm, class_names, str(output_path), class_order)
            plot_confusion_matrix(cm, class_names, f"{param_name} / {group} / accumulated", 
                                str(output_path), class_order)
        
        # total
        total_cm = compute_cm_from_judgment_results(judgment_results, n_classes)
        output_path = output_param_dir / "total"
        save_confusion_matrix_csv(total_cm, class_names, str(output_path), class_order)
        plot_confusion_matrix(total_cm, class_names, f"{param_name} / total", 
                            str(output_path), class_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # configファイルの読み込み
    cfg = load_config(args.config)

    eval_result_dir = Path(cfg.eval_result_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # config.lock.yamlの保存
    shutil.copy(args.config, output_dir / "analysis_config.lock.yaml")

    sample_groups = cfg.sample_groups
    modes = cfg.mode  # リスト形式
    class_names = cfg.class_names
    class_order = getattr(cfg, 'class_order', None)

    # 最初のサンプル群からパラメータディレクトリ一覧を取得
    first_group_dir = eval_result_dir / sample_groups[0]
    param_dirs = list_param_dirs(first_group_dir)

    # modeループ
    for mode in modes:
        print(f"Processing mode: {mode}")

        # 第1段階: 判定結果の計算と保存
        param_judgment_results = compute_and_save_judgments(
            param_dirs, sample_groups, mode, eval_result_dir, output_dir
        )

        # 第2段階: 混同行列の計算とプロット
        compute_and_save_confusion_matrices(
            param_dirs, sample_groups, param_judgment_results, 
            output_dir, class_names, class_order
        )

        # 第3段階: メトリクスのプロット
        ylim = getattr(cfg, 'plot_ylim', [0, 1])
        plot_performance_summary(param_dirs, output_dir, param_key="Nx", ylim=ylim)

    print("Analysis finished")


if __name__ == "__main__":
    main()
