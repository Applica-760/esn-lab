from pathlib import Path
from itertools import product
import pandas as pd

def load_10fold_csv_mapping(csv_dir: Path) -> dict[str, Path]:
    """10分割されたCSVファイルのマッピングを読み込む ('a' -> Path(...), ...)。"""
    letters = [chr(c) for c in range(ord("a"), ord("j") + 1)]
    mapping: dict[str, Path] = {}
    for ch in letters:
        p = csv_dir / f"10fold_{ch}.csv"
        if not p.exists():
            raise FileNotFoundError(f"10-fold csv not found: {p}")
        mapping[ch] = p
    return mapping

def read_data_from_csvs(paths: list[Path]) -> tuple[list[str], list[str], list[int]]:
    """指定された複数のCSVから学習データを読み込む。"""
    ids: list[str] = []
    img_paths: list[str] = []
    class_ids: list[int] = []

    usecols = ["file_path", "behavior"]
    for p in paths:
        df = pd.read_csv(p, usecols=usecols)
        if list(df.columns) != usecols:
            raise ValueError(f"CSV columns mismatch at {p}. expected={usecols}, got={list(df.columns)}")
        
        for fp, beh in zip(df["file_path"], df["behavior"]):
            fp_str = str(fp)
            ids.append(Path(fp_str).stem)
            img_paths.append(fp_str)
            class_ids.append(int(beh))
            
    return ids, img_paths, class_ids

def flatten_search_space(ss: dict[str, list] | None) -> list[tuple[dict, str]]:
    """
    ハイパーパラメータの検索空間(辞書)を、
    (パラメータ辞書, パラメータタグ文字列)のリストに変換する。
    """
    if not ss:
        return [({}, "default")]

    items = []
    for k, vals in ss.items():
        if not k.startswith("model."):
            raise ValueError(f"search_space key must start with 'model.': {k}")
        field = k.split(".", 1)[1]
        items.append((field, list(vals)))

    keys = [k for k, _ in items]
    lists = [v for _, v in items]
    
    combos = []
    for values in product(*lists):
        d = {k: v for k, v in zip(keys, values)}
        tag = "_".join([f"{k}={v}" for k, v in d.items()])
        combos.append((d, tag))
        
    return combos