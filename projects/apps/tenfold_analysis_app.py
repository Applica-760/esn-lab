import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from projects.utils.io.config import load_config
from projects.utils.compute.confusion import (
    compute_cm_from_judgment_results,
    save_all_total_cms,
)
from projects.utils.io.confusion import (
    save_confusion_matrix_csv,
    load_confusion_matrix_csv,
)
from projects.utils.plot.plot_confusion import plot_confusion_matrix
from projects.utils.io.results import (
    load_eval_results,
    save_judgment_results,
)
from projects.utils.compute.judgment import compute_judgment_results
from projects.utils.io.weights import list_param_dirs
from projects.utils.plot.plot_performance import plot_metric_by_param, plot_performance_summary


"""
python -m projects.apps.tenfold_analysis_app --config projects/configs/tenfold_analysis.yaml
"""


def one_process(param_dir, group, mode, eval_result_dir, output_dir, class_names, class_order):
    """
    単一パラメータ × 単一サンプル群の解析を実行
    
    Returns:
        判定結果のリスト（スキップ時はNone）
    """
    param_name = param_dir.name
    n_classes = len(class_names)
    json_path = eval_result_dir / group / param_name / f"{mode}_results.json"
    output_param_dir = output_dir / param_name

    # スキップ判定
    accumulated_path = output_param_dir / group / "accumulated.csv"
    if accumulated_path.exists():
        print(f"skipped (already exists): {param_name} group={group} mode={mode}")
        return None

    if not json_path.exists():
        print(f"skipped (not found): {param_name} group={group} mode={mode}")
        return None

    # 評価結果を読み込み
    eval_results = load_eval_results(str(json_path))

    # 判定結果を計算
    judgment_results = compute_judgment_results(eval_results, group=group)

    # 判定結果からfold単位の混同行列を計算
    fold_indices = sorted(set(r["fold_index"] for r in judgment_results))
    fold_cms = [
        compute_cm_from_judgment_results(judgment_results, n_classes, fold_index=i, group=group)
        for i in fold_indices
    ]

    # fold単位の混同行列を保存
    for i, cm in enumerate(fold_cms):
        output_path = output_param_dir / group / f"fold_{i}"
        save_confusion_matrix_csv(cm, class_names, str(output_path), class_order)
        plot_confusion_matrix(cm, class_names, f"{param_name} / {group} / fold {i}", str(output_path), class_order)

    # サンプル群累計の混同行列を保存
    group_accumulated_cm = np.sum(fold_cms, axis=0)
    output_path = output_param_dir / group / "accumulated"
    save_confusion_matrix_csv(group_accumulated_cm, class_names, str(output_path), class_order)
    plot_confusion_matrix(group_accumulated_cm, class_names, f"{param_name} / {group} / accumulated", str(output_path), class_order)

    print(f"proceed: {param_name} group={group} mode={mode}")
    return {"param_name": param_name, "judgment_results": judgment_results}


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
        # パラメータごとの判定結果を蓄積する辞書
        param_judgment_results = {param_dir.name: [] for param_dir in param_dirs}

        for group in sample_groups:
            print(f"mode={mode} group={group}")

            # パラメータごとに並列処理
            with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                futures = [
                    executor.submit(one_process, param_dir, group, mode, eval_result_dir, output_dir, class_names, class_order)
                    for param_dir in param_dirs
                ]
                for future in futures:
                    result = future.result()
                    if result is not None:
                        param_name = result["param_name"]
                        param_judgment_results[param_name].extend(result["judgment_results"])

        # 判定結果をパラメータ組ごとに保存
        print(f"Saving judgment results for mode={mode}")
        for param_dir in param_dirs:
            param_name = param_dir.name
            judgment_results = param_judgment_results[param_name]
            if judgment_results:
                output_path = output_dir / param_name / "judgment_results"
                save_judgment_results(judgment_results, str(output_path))
                print(f"Judgment results saved: {param_name}")

        # 全サンプル群統合の混同行列を計算（並列処理後）
        print(f"Computing total confusion matrices for mode={mode}")
        save_all_total_cms(param_dirs, sample_groups, output_dir, class_names, class_order)

        # 性能指標のサマリープロット
        print(f"Plotting performance summary for mode={mode}")
        ylim = getattr(cfg, 'plot_ylim', [0, 1])
        plot_performance_summary(param_dirs, output_dir, param_key="Nx", ylim=ylim)

    print("Analysis finished")


if __name__ == "__main__":
    main()
