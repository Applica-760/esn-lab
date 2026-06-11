"""
アプリケーション初期化ユーティリティ
"""

import argparse
import shutil
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def load_config(config_path: str) -> SimpleNamespace:
    """
    YAML設定ファイルをロード
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    print("config loaded")
    return SimpleNamespace(**data)


def setup_app_environment() -> tuple[SimpleNamespace, Path]:
    """
    アプリケーション環境のセットアップ
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(getattr(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.lock.yaml")
    return cfg, output_dir


def setup_task_environment(module_path: str, config_name: str | None = None) -> SimpleNamespace:
    """
    モジュールパスから対応するcfg.yamlを読み込み、設定を返す。
    config_name が指定された場合はタスクディレクトリ内の同名ファイルを優先する。
    """
    # "projects.tasks.train.app" -> "projects/tasks/train"
    parts = module_path.split(".")
    task_dir = Path("/".join(parts[:-1]))
    if config_name is not None:
        config_path = task_dir / config_name
    else:
        config_path = task_dir / "cfg.yaml"

    cfg = load_config(str(config_path))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(config_path), output_dir / "config.lock.yaml")
    cfg.output_dir = output_dir

    return cfg


def tenfold_data_loader(dataset_dir: str | Path):
    """
    10fold用のデータセットをロード
    """
    dataset_dir = Path(dataset_dir)
    npz_files = sorted(dataset_dir.glob("*.npz"))

    # 全foldのデータを格納する3つのリスト
    data_folds = []
    label_folds = []
    id_folds = []

    # 各NPZファイルを処理
    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=True) as npz_data:
            num_samples = int(npz_data["num_samples"])

            # 各サンプルのデータをリストに格納
            data_list = []
            label_list = []
            id_list = []

            for i in range(num_samples):
                data_list.append(npz_data[f"{i}_data"])
                label_list.append(npz_data[f"{i}_label"])
                id_list.append(str(npz_data[f"{i}_id"]))

            # 各リストに追加
            data_folds.append(data_list)
            label_folds.append(label_list)
            id_folds.append(id_list)

    print("dataset loaded")

    return data_folds, label_folds, id_folds


def build_param_grid(cfg):
    """
    パラメータグリッドを構築
    """
    param_names = ["Nx", "input_scale", "density", "rho"]
    param_values = [cfg.Nx, cfg.input_scale, cfg.density, cfg.rho]

    if getattr(cfg, "beta_auto", False):
        param_names.append("beta_scale")
        param_values.append(cfg.beta_scale)

    # 全組み合わせを生成
    combinations = product(*param_values)

    print("grid params built")

    # 辞書のリストに変換
    return [dict(zip(param_names, combo)) for combo in combinations]
