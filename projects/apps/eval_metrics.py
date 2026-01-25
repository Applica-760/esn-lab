import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

from projects.utils.app_init import setup_app_environment
from projects.utils.weights import list_param_dirs
from projects.utils.eval.confusion import compute_cm_from_judgment_results, save_confusion_matrix
from projects.utils.eval.filter import filter_by_group, filter_by_fold
from projects.utils.eval.judgment import load_judgment_results
from projects.utils.eval.metrics import plot_performance_summary


"""
python -m projects.apps.eval_metrics --config projects/configs/eval_metrics.yaml
"""


def load_judgment_results_for_params(param_dirs, mode, judge_dir):
    """
    保存された判定結果の読み込み
    """
    param_judgment_results = {}

    for param_dir in param_dirs:
        param_name = param_dir.name
        judgment_csv_path = judge_dir / param_name / f"judgment_results_{mode}.csv"
        
        if not judgment_csv_path.exists():
            print(f"  Warning: Judgment results not found: {judgment_csv_path}")
            param_judgment_results[param_name] = []
            continue
        
        judgment_results = load_judgment_results(judgment_csv_path)
        param_judgment_results[param_name] = judgment_results

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
    judge_dir = Path(cfg.judge_dir)

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

        # 判定結果の読み込み
        param_judgment_results = load_judgment_results_for_params(
            param_dirs, mode, judge_dir
        )

        # 混同行列の計算とプロット
        compute_and_save_confusion_matrices(
            param_dirs, sample_groups, param_judgment_results, 
            output_dir, class_names, class_order, mode
        )

        # メトリクスのプロット
        ylim = getattr(cfg, 'plot_ylim', [0, 1])
        plot_performance_summary(param_dirs, output_dir, param_key="Nx", ylim=ylim, mode=mode)

    print("evaluation finished")


if __name__ == "__main__":
    main()
