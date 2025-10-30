from pathlib import Path
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
    """指定された複数のCSVから学習/評価データを読み込む。"""
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
