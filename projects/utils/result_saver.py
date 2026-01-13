import os
import json
import numpy as np
from pathlib import Path


def load_eval_results(path: str) -> list:
    """
    評価結果のJSONファイルをロード
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_eval_results(results: list, output_path: str) -> None:
    """
    評価結果をJSON形式で保存
    """
    # ndarrayをリストに変換
    serializable_results = _convert_to_serializable(results)
    
    # ディレクトリがなければ作成
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def _convert_to_serializable(obj):
    """
    任意のオブジェクトを再帰的にJSON serializable形式に変換
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj
