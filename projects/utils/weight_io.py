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


def list_param_dirs(fold_dir: str) -> list:
    """
    fold_dir内のパラメータディレクトリ一覧を返す
    """
    fold_path = Path(fold_dir)

    return [d for d in fold_path.iterdir() if d.is_dir()]
    
