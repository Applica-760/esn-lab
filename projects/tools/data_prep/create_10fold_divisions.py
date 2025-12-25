#!/usr/bin/env python3
"""
IDベースで10-fold + 各foldの10分割を行い、npzファイルも生成

Usage:
python projects/tools/data_prep/create_10fold_divisions.py \
  --csv data/get_300seqs.csv \
  --output-csv dataset/10fold_ids \
  --output-npz dataset/10fold_npy \
  --images-root data/all_s \
  --min_width 8000 \
  --seeds_base 20251005
"""
import argparse
import string
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd


def extract_uniform_label(seq: str) -> int:
    uniq = set(seq)
    if len(uniq) != 1 or next(iter(uniq)) not in {"0", "1", "2"}:
        raise ValueError(f"Invalid sequence: {seq}")
    return int(next(iter(uniq)))


def extract_image_id(file_path: str) -> str:
    stem = Path(file_path).stem
    return stem[:-7] if stem.endswith("_ffmpeg") else stem


def build_balanced_10fold(df: pd.DataFrame, seeds_base: int) -> Dict[str, pd.DataFrame]:
    tmp = df.copy()
    tmp["behavior"] = [extract_uniform_label(s) for s in df["converted_300"]]
    tmp["image_id"] = tmp["file_path"].apply(extract_image_id)
    
    min_num = int(tmp["behavior"].value_counts().min())
    suffixes = list(string.ascii_lowercase[:10])
    folds = {}
    
    for i, fold_id in enumerate(suffixes):
        seed_i = seeds_base + i
        rng = np.random.default_rng(seed=seed_i)
        parts: List[pd.DataFrame] = []
        
        for lab in [0, 1, 2]:
            sub = tmp[tmp["behavior"] == lab]
            idx = rng.choice(sub.index.to_numpy(), size=min_num, replace=False)
            parts.append(sub.loc[idx])
        
        balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed_i).reset_index(drop=True)
        folds[fold_id] = balanced[["image_id", "behavior"]].copy()
    
    return folds


def divide_fold_into_10(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df_sorted = df.sort_values(by=["behavior", "image_id"]).reset_index(drop=True)
    class_samples = {cls: df_sorted[df_sorted["behavior"] == cls].reset_index(drop=True) for cls in [0, 1, 2]}
    
    folds_per_class = {}
    for cls in [0, 1, 2]:
        samples = class_samples[cls]
        groups = []
        idx = 0
        
        for _ in range(5):
            groups.append(samples.iloc[idx:idx+9])
            idx += 9
        for _ in range(5):
            groups.append(samples.iloc[idx:idx+8])
            idx += 8
        
        folds_per_class[cls] = groups
    
    fold_chars = list(string.ascii_lowercase[:10])
    divisions = {}
    
    for fold_idx in range(10):
        fold_parts = [folds_per_class[cls][fold_idx] for cls in [0, 1, 2]]
        combined = pd.concat(fold_parts, axis=0).reset_index(drop=True)
        divisions[fold_chars[fold_idx]] = combined[["image_id", "behavior"]].copy()
    
    return divisions


def load_image_as_array(image_id: str, images_root: Path) -> np.ndarray:
    img_path = images_root / f"{image_id}_ffmpeg.jpg"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img.T  # 転置して(width, height)の形状にする


def create_onehot_label(behavior: int, width: int) -> np.ndarray:
    label = np.zeros((3, width), dtype=np.float32)
    label[behavior, :] = 1.0
    return label


def save_divisions(main_fold_id: str, divisions: Dict[str, pd.DataFrame], csv_dir: Path = None, npz_dir: Path = None, images_root: Path = None):
    if csv_dir is not None:
        fold_csv_dir = csv_dir / main_fold_id
        fold_csv_dir.mkdir(parents=True, exist_ok=True)
    
    if npz_dir is not None:
        fold_npz_dir = npz_dir / main_fold_id
        fold_npz_dir.mkdir(parents=True, exist_ok=True)
    
    for div_id, df in divisions.items():
        if csv_dir is not None:
            df.to_csv(fold_csv_dir / f"fold_{div_id}.csv", index=False)
        
        if npz_dir is not None and images_root is not None:
            npz_data = {}
            for i, row in df.iterrows():
                img_data = load_image_as_array(row["image_id"], images_root)
                label = create_onehot_label(row["behavior"], img_data.shape[0])
                npz_data[f"{i}_id"] = row["image_id"]
                npz_data[f"{i}_data"] = img_data
                npz_data[f"{i}_label"] = label.T  # 転置して(時系列長, クラス数)の形状にする
            npz_data["num_samples"] = len(df)
            np.savez_compressed(fold_npz_dir / f"fold_{div_id}.npz", **npz_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-npz", type=str, default=None)
    parser.add_argument("--images-root", type=str, default=None)
    parser.add_argument("--min_width", type=int, default=8000)
    parser.add_argument("--seeds_base", type=int, default=20251005)
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv, dtype={"converted_300": str})
    cond = (df["uniform_flag"] == 1) & (df["image_width_px"] >= args.min_width)
    df_filtered = df.loc[cond].reset_index(drop=True)
    
    folds_stage1 = build_balanced_10fold(df_filtered, args.seeds_base)
    
    csv_dir = Path(args.output_csv) if args.output_csv else None
    npz_dir = Path(args.output_npz) if args.output_npz else None
    images_root = Path(args.images_root) if args.images_root else None
    
    for main_fold_id, fold_df in folds_stage1.items():
        divisions = divide_fold_into_10(fold_df)
        save_divisions(main_fold_id, divisions, csv_dir, npz_dir, images_root)


if __name__ == "__main__":
    main()
