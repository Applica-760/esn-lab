import json
from pathlib import Path

import numpy as np


def build_param_str(params: dict) -> str:
    """
    パラメータ辞書からディレクトリ名用の文字列を生成
    """
    return (
        f"Nx{params['Nx']}_dens{params['density']}_inscl{params['input_scale']}_rho{params['rho']}"
    )


def is_valid_weight_file(filepath: str) -> bool:
    """
    重みファイルが存在し、破損していないかを確認
    """
    path = Path(filepath)
    if not path.exists():
        return False

    try:
        with np.load(path) as data:
            if "weight" not in data:
                return False
            _ = data["weight"]
        return True
    except Exception:
        return False


def save_single_weight(params: dict, weight: np.ndarray, fold_idx: int, output_dir: str) -> None:
    """
    単一のfoldの重みを保存
    """
    param_dir = Path(output_dir) / build_param_str(params)
    param_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = param_dir / "metadata.json"
    if not metadata_path.exists():
        with open(metadata_path, "w") as f:
            json.dump(params, f, indent=2)

    np.savez(param_dir / f"fold{fold_idx}.npz", weight=weight)


def save_tenfold_weights(params: dict, weights_list: list, output_dir: str) -> None:
    """
    10fold訓練の重みを保存
    """
    param_dir = Path(output_dir) / build_param_str(params)
    param_dir.mkdir(parents=True, exist_ok=True)

    with open(param_dir / "metadata.json", "w") as f:
        json.dump(params, f, indent=2)

    for i, weight in enumerate(weights_list):
        np.savez(param_dir / f"fold{i}.npz", weight=weight)


def load_tenfold_weights(param_dir: str) -> list:
    """
    パラメータディレクトリから10個の重みをロード
    """
    param_dir = Path(param_dir)
    weights_list = []
    for i in range(10):
        data = np.load(param_dir / f"fold{i}.npz")
        weights_list.append(data["weight"])
    return weights_list


def load_metadata(param_dir: str) -> dict:
    """
    metadata.jsonを読み込み
    """
    with open(Path(param_dir) / "metadata.json", "r") as f:
        return json.load(f)


def list_param_dirs(fold_dir: str) -> list:
    """
    fold_dir内のパラメータディレクトリ一覧を返す
    """
    return [d for d in Path(fold_dir).iterdir() if d.is_dir()]
