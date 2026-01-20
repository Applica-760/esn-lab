import os
import json
import csv
import numpy as np
from pathlib import Path


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


def save_judgment_results(judgment_results: list, output_path: str) -> None:
    """
    判定結果をCSV形式で保存
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = str(output_path) + ".csv"

    fieldnames = ["group", "fold_index", "id", "pred_label", "true_label", "is_correct"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judgment_results)


def load_judgment_results(csv_path: str) -> list:
    """
    判定結果CSVを読み込み、辞書のリストとして返す
    """
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "group": row["group"],
                "fold_index": int(row["fold_index"]),
                "id": row["id"],
                "pred_label": int(row["pred_label"]),
                "true_label": int(row["true_label"]),
                "is_correct": row["is_correct"] == "True",
            })

    return results
