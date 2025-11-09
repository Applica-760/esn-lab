# utils/io
import json
import os
import uuid
import numpy as np
from pathlib import Path
from dataclasses import asdict, is_dataclass

from esn_lab.setup.config import TargetOutput, TargetOutputData

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
    
 
def save_numpy_npy_atomic(arr: np.ndarray, save_dir: str | Path, file_name: str) -> Path:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / file_name
    tmp = dst.with_suffix(dst.suffix + f".{uuid.uuid4().hex}.tmp")

    # Write to a temp file handle to avoid numpy adding extensions implicitly
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    finally:
        # If something went wrong and tmp still exists, attempt cleanup
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return dst
