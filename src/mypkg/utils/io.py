# utils/io
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict, is_dataclass

from mypkg.utils.config import TargetOutput, TargetOutputData

# データを再帰的に走査し，mdataclass, numpy配列をjson書き込みできるように変換
def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    return obj


# TargetOutputのidフィールドを，辞書のキーにする
def to_keyed_dict(obj):
    if isinstance(obj, TargetOutput):
        return {obj.id: to_jsonable(obj.data)}
    

# json保存したTargetOutput dataclassに復元
def target_output_from_dict(d: dict) -> TargetOutput:
    data_dict = d.get("data")
    data = TargetOutputData(**data_dict) if data_dict else None
    return TargetOutput(id=d.get("id"), data=data)


# 実行結果をjsonに保存
def save_json(results: dict, save_dir, file_name):
    with open(Path(save_dir) / Path(file_name), "w", encoding="utf-8") as f:
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()  # numpy配列をlistに変換
            record = {"id": key, "data": value}
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
    return

# jsonを読み出し
def load_jsonl(saved_path):
    path = Path(saved_path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    