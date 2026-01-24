import os
import json
import numpy as np


def is_valid_result_file(filepath: str) -> bool:
    """
    結果ファイルが存在し、破損していないかを確認
    """
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            # リスト形式であることを確認
            if not isinstance(data, list):
                return False
        return True
    except Exception:
        return False


def save_pred_results(results: list, output_path: str) -> None:
    """
    予測結果をJSON形式で保存
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
