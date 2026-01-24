import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path

from projects.utils.app_init import setup_app_environment
from projects.utils.weights import list_param_dirs
from projects.utils.eval.confusion import compute_cm_from_judgment_results, save_confusion_matrix
from projects.utils.eval.filter import filter_by_group, filter_by_fold
from projects.utils.eval.judgment import compute_judgment_results, save_judgment_results
from projects.utils.eval.metrics import plot_performance_summary


"""
python -m projects.apps.eval_metrics --config projects/configs/eval_metrics.yaml
"""


def compute_and_save_judgments(param_dirs, sample_groups, mode, pred_result_dir, output_dir):
    """
    判定結果の計算と保存
    """
    param_judgment_results = {param_dir.name: [] for param_dir in param_dirs}

    for param_dir in param_dirs:
        param_name = param_dir.name
        
        # すべてのgroupを処理
        for group in sample_groups:
            json_path = pred_result_dir / group / param_name / f"{mode}_results.json"

            if not json_path.exists():
                continue

            with open(json_path, 'r') as f:
                pred_results = json.load(f)
            judgment_results = compute_judgment_results(pred_results, group=group)
            param_judgment_results[param_name].extend(judgment_results)
        
        # このparam_dirのすべてのgroupを処理し終えたので保存
        if param_judgment_results[param_name]:
            output_path = output_dir / param_name / f"judgment_results_{mode}"
            save_judgment_results(param_judgment_results[param_name], output_path)

    return param_judgment_results


def compute_and_save_confusion_matrices(param_dirs, sample_groups, param_judgment_results, 
                                        output_dir, class_names, class_order, mode):
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
            group_judgment_results = filter_by_group(judgment_results, [group])
            if not group_judgment_results:
                continue
            
            # fold別
            fold_indices = sorted(set(r["fold_index"] for r in group_judgment_results))
            for fold_idx in fold_indices:
                fold_judgment_results = filter_by_fold(group_judgment_results, [fold_idx])
                cm = compute_cm_from_judgment_results(fold_judgment_results, n_classes)
                output_path = output_param_dir / group / f"fold_{fold_idx}_{mode}"
                save_confusion_matrix(cm, class_names, f"{param_name} / {group} / fold {fold_idx} ({mode})",
                                    output_path, class_order)
            
            # accumulated
            cm = compute_cm_from_judgment_results(group_judgment_results, n_classes)
            output_path = output_param_dir / group / f"accumulated_{mode}"
            save_confusion_matrix(cm, class_names, f"{param_name} / {group} / accumulated ({mode})",
                                output_path, class_order)
        
        # total
        total_cm = compute_cm_from_judgment_results(judgment_results, n_classes)
        output_path = output_param_dir / f"total_{mode}"
        save_confusion_matrix(total_cm, class_names, f"{param_name} / total ({mode})",
                            output_path, class_order)


def main():
    cfg, output_dir = setup_app_environment()

    pred_result_dir = Path(cfg.pred_result_dir)

    sample_groups = cfg.sample_groups
    modes = cfg.mode  # リスト形式
    class_names = cfg.class_names
    class_order = getattr(cfg, 'class_order', None)

    # 最初のサンプル群からパラメータディレクトリ一覧を取得
    first_group_dir = pred_result_dir / sample_groups[0]
    param_dirs = list_param_dirs(first_group_dir)

    # modeループ
    for mode in modes:
        print(f"Processing mode: {mode}")

        # 第1段階: 判定結果の計算と保存
        param_judgment_results = compute_and_save_judgments(
            param_dirs, sample_groups, mode, pred_result_dir, output_dir
        )

        # 第2段階: 混同行列の計算とプロット
        compute_and_save_confusion_matrices(
            param_dirs, sample_groups, param_judgment_results, 
            output_dir, class_names, class_order, mode
        )

        # 第3段階: メトリクスのプロット
        ylim = getattr(cfg, 'plot_ylim', [0, 1])
        plot_performance_summary(param_dirs, output_dir, param_key="Nx", ylim=ylim, mode=mode)

    print("evaluation finished")


if __name__ == "__main__":
    main()
