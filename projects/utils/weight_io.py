import os
import json
import numpy as np
from pathlib import Path


def build_param_str(params: dict) -> str:
    """
    パラメータ辞書からディレクトリ名用の文字列を生成
    """
    return f"Nx{params['Nx']}_dens{params['density']}_inscl{params['input_scale']}_rho{params['rho']}"


def is_valid_weight_file(filepath: str) -> bool:
    """
    重みファイルが存在し、破損していないかを確認
    
    Args:
        filepath: 確認対象の.npzファイルパス
    
    Returns:
        ファイルが存在し、正常に読み込める場合True
    """
    if not os.path.exists(filepath):
        return False
    
    try:
        with np.load(filepath) as data:
            # 'weight' キーが存在し、配列として読み込めるか確認
            if 'weight' not in data:
                return False
            _ = data['weight']
        return True
    except Exception:
        return False


def save_single_weight(params: dict, weight: np.ndarray, fold_idx: int, output_dir: str) -> None:
    """
    単一のfoldの重みを保存
    """
    # パラメータごとのサブディレクトリを作成
    param_str = build_param_str(params)
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)
    
    # metadata.jsonが存在しない場合のみ保存
    metadata_path = os.path.join(param_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    # 指定されたfoldの重みを保存
    filename = f"fold{fold_idx}.npz"
    filepath = os.path.join(param_dir, filename)
    np.savez(filepath, weight=weight)
    
    return

def save_tenfold_weights(params: dict, weights_list: list, output_dir: str) -> None:
    """
    10fold訓練の重みを保存
    """
    # パラメータごとのサブディレクトリを作成
    param_str = build_param_str(params)
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

    return


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


def list_param_dirs(fold_dir: str, target_params=None) -> list:
    """
    fold_dir内のパラメータディレクトリ一覧を返す

    Args:
        fold_dir: パラメータディレクトリを含む親ディレクトリ
        target_params: None（全パラメータ）、文字列（単一）、またはリスト（複数）

    Returns:
        パラメータディレクトリのPathリスト
    """
    fold_path = Path(fold_dir)

    if target_params is None:
        # 全パラメータディレクトリ
        return [d for d in fold_path.iterdir() if d.is_dir()]
    elif isinstance(target_params, str):
        # 単一パラメータ
        target_path = fold_path / target_params
        if target_path.is_dir():
            return [target_path]
        else:
            raise FileNotFoundError(f"Parameter directory not found: {target_path}")
    elif isinstance(target_params, list):
        # 複数パラメータ
        result = []
        for param in target_params:
            target_path = fold_path / param
            if target_path.is_dir():
                result.append(target_path)
            else:
                raise FileNotFoundError(f"Parameter directory not found: {target_path}")
        return result
    else:
        raise ValueError(f"target_params must be None, str, or list, got {type(target_params)}")
