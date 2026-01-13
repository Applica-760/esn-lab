import argparse
import shutil
from pathlib import Path

from projects.utils.config_loader import load_config
from projects.utils.result_saver import load_eval_results
from projects.utils.vote import majority_vote_prediction, majority_vote_label
from projects.utils.confusion import (
    compute_confusion_matrix,
    accumulate_confusion_matrices,
    save_confusion_matrix,
)
from projects.utils.weight_io import list_param_dirs


"""
python -m projects.apps.tenfold_analysis_app --config projects/configs/tenfold_analysis.yaml
"""


def process_single_json(json_path: str, n_classes: int) -> list:
    """
    単一のJSONファイル（10fold分の結果）を処理し、fold単位の混同行列リストを返す

    Args:
        json_path: eval結果のJSONファイルパス
        n_classes: クラス数

    Returns:
        fold単位の混同行列リスト (長さ10)
    """
    data = load_eval_results(json_path)
    fold_cms = []

    for fold_data in data:
        y_true = []
        y_pred = []

        for sample in fold_data["results"]:
            pred_label = majority_vote_prediction(sample["predictions"])
            true_label = majority_vote_label(sample["labels"])
            y_pred.append(pred_label)
            y_true.append(true_label)

        cm = compute_confusion_matrix(y_true, y_pred, n_classes)
        fold_cms.append(cm)

    return fold_cms


def process_single_param(param_dir: Path, sample_groups: list, mode: str,
                         class_names: list, output_base: Path, eval_result_dir: Path,
                         class_order: list = None) -> None:
    """
    単一パラメータの全サンプル群を処理（将来の並列化用に切り出し）

    Args:
        param_dir: パラメータディレクトリ（eval_result_dir直下のパラメータ名ディレクトリ）
        sample_groups: サンプル群リスト ["a", "b", ..., "j"]
        mode: "test" or "train"
        class_names: クラス名リスト
        output_base: 出力ベースディレクトリ
        eval_result_dir: eval結果のベースディレクトリ
        class_order: 表示順に対応する元のインデックスリスト
    """
    param_name = param_dir.name
    n_classes = len(class_names)
    output_param_dir = output_base / param_name

    all_group_cms = []  # 全サンプル群の混同行列を蓄積

    for group in sample_groups:
        json_path = eval_result_dir / group / param_name / f"{mode}_results.json"

        if not json_path.exists():
            print(f"Skip: {json_path} not found")
            continue

        # fold単位の混同行列を取得
        fold_cms = process_single_json(str(json_path), n_classes)

        # fold単位の混同行列を保存
        for i, cm in enumerate(fold_cms):
            output_path = output_param_dir / group / f"fold_{i}"
            save_confusion_matrix(cm, class_names, f"{param_name} / {group} / fold {i}", str(output_path), class_order)

        # サンプル群累計の混同行列を保存
        group_accumulated_cm = accumulate_confusion_matrices(fold_cms)
        output_path = output_param_dir / group / "accumulated"
        save_confusion_matrix(group_accumulated_cm, class_names,
                              f"{param_name} / {group} / accumulated", str(output_path), class_order)

        all_group_cms.append(group_accumulated_cm)
        print(f"Processed: {param_name} / {group}")

    # 全サンプル群統合の混同行列を保存
    if all_group_cms:
        total_cm = accumulate_confusion_matrices(all_group_cms)
        output_path = output_param_dir / "total"
        save_confusion_matrix(total_cm, class_names, f"{param_name} / total", str(output_path), class_order)
        print(f"Total confusion matrix saved: {param_name}")


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
    mode = cfg.mode
    class_names = cfg.class_names
    class_order = getattr(cfg, 'class_order', None)
    target_params = getattr(cfg, 'target_params', None)

    # 最初のサンプル群からパラメータディレクトリ一覧を取得
    first_group_dir = eval_result_dir / sample_groups[0]
    param_dirs = list_param_dirs(first_group_dir, target_params)

    # パラメータごとに処理
    for param_dir in param_dirs:
        process_single_param(param_dir, sample_groups, mode, class_names, output_dir, eval_result_dir, class_order)

    print("Analysis finished")


if __name__ == "__main__":
    main()
