import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from projects.utils.config_loader import load_config
from projects.utils.confusion import (
    accumulate_confusion_matrices,
    save_confusion_matrix,
    compute_cm_from_eval_results,
    save_all_total_cms,
)
from projects.utils.weight_io import list_param_dirs


"""
python -m projects.apps.tenfold_analysis_app --config projects/configs/tenfold_analysis.yaml
"""


def one_process(param_dir, group, mode, eval_result_dir, output_dir, class_names, class_order):
    """
    単一パラメータ × 単一サンプル群の解析を実行
    """
    param_name = param_dir.name
    n_classes = len(class_names)
    json_path = eval_result_dir / group / param_name / f"{mode}_results.json"
    output_param_dir = output_dir / param_name

    # スキップ判定
    accumulated_path = output_param_dir / group / "accumulated.csv"
    if accumulated_path.exists():
        print(f"skipped (already exists): {param_name} group={group} mode={mode}")
        return

    if not json_path.exists():
        print(f"skipped (not found): {param_name} group={group} mode={mode}")
        return

    # fold単位の混同行列を取得
    fold_cms = compute_cm_from_eval_results(str(json_path), n_classes)

    # fold単位の混同行列を保存
    for i, cm in enumerate(fold_cms):
        output_path = output_param_dir / group / f"fold_{i}"
        save_confusion_matrix(cm, class_names, f"{param_name} / {group} / fold {i}", str(output_path), class_order)

    # サンプル群累計の混同行列を保存
    group_accumulated_cm = accumulate_confusion_matrices(fold_cms)
    output_path = output_param_dir / group / "accumulated"
    save_confusion_matrix(group_accumulated_cm, class_names,
                          f"{param_name} / {group} / accumulated", str(output_path), class_order)

    print(f"proceed: {param_name} group={group} mode={mode}")
    return


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
        for group in sample_groups:
            print(f"mode={mode} group={group}")

            # パラメータごとに並列処理
            with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                futures = [
                    executor.submit(one_process, param_dir, group, mode, eval_result_dir, output_dir, class_names, class_order)
                    for param_dir in param_dirs
                ]
                for future in futures:
                    future.result()

        # 全サンプル群統合の混同行列を計算（並列処理後）
        print(f"Computing total confusion matrices for mode={mode}")
        save_all_total_cms(param_dirs, sample_groups, output_dir, class_names, class_order)

    print("Analysis finished")


if __name__ == "__main__":
    main()
