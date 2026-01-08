import os
import json
import numpy as np
from pathlib import Path


def save_tenfold_weights(params: dict, weights_list: list, output_dir: str) -> None:
    """
    10fold訓練の重みを保存
    """
    # パラメータごとのサブディレクトリを作成
    param_str = f"Nx{params['Nx']}_dens{params['density']}_inscl{params['input_scale']}_rho{params['rho']}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)
    
    # パラメータをmetadata.jsonとして保存
    metadata_path = os.path.join(param_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    # 各foldの重みを個別に保存
    for i, weight in enumerate(weights_list):
        filename = f"fold{i}.npz"
        filepath = os.path.join(param_dir, filename)
        np.savez(filepath, weight=weight)


def load_tenfold_weights(param_dir: str) -> list:
    """
    パラメータディレクトリから10個の重みをロード
    """
    weights_list = []
    for i in range(10):
        filepath = os.path.join(param_dir, f"fold{i}.npz")
        data = np.load(filepath)
        weights_list.append(data["weight"])
    return weights_list


def load_metadata(param_dir: str) -> dict:
    """
    metadata.jsonを読み込み
    """
    metadata_path = os.path.join(param_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)


def list_param_dirs(fold_dir: str, target_params: str = None) -> list:
    """
    fold_dir内のパラメータディレクトリ一覧を返す
    """
    fold_path = Path(fold_dir)
    
    if target_params is not None:
        # 特定のパラメータのみ
        target_path = fold_path / target_params
        if target_path.is_dir():
            return [target_path]
        else:
            raise FileNotFoundError(f"Parameter directory not found: {target_path}")
    else:
        # 全パラメータディレクトリ
        return [d for d in fold_path.iterdir() if d.is_dir()]
