#!/usr/bin/env python3
"""
10-fold CSV + 画像データを NPY 形式に変換する前処理スクリプト

Usage:
    python tools/dataprep/convert_10fold_to_npy.py \
        --csv-dir dataset/10fold_csvs/ \
        --output-dir dataset/10fold_npy
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Convert 10-fold CSV + images to NPY format for faster loading"
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        required=True,
        help="Directory containing 10fold_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for NPY files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "npy_per_sample"],
        default="npz",
        help="Output format: 'npz' (compressed ragged array per fold) or 'npy_per_sample' (one .npy per sample)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if output directory already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - validate inputs without writing output",
    )
    return parser.parse_args()


def validate_inputs(csv_dir: Path) -> List[str]:
    """入力ディレクトリの検証と利用可能なfoldの検出
    
    Returns:
        利用可能なfold ID一覧（例: ['a', 'b', 'c', ...]）
    """
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    
    available_folds = []
    for ch in [chr(c) for c in range(ord("a"), ord("j") + 1)]:
        csv_path = csv_dir / f"10fold_{ch}.csv"
        if csv_path.exists():
            available_folds.append(ch)
    
    if not available_folds:
        raise ValueError(f"No 10fold_*.csv files found in {csv_dir}")
    
    print(f"[INFO] Found {len(available_folds)} folds: {available_folds}")
    return available_folds


def load_fold_data(csv_path: Path) -> Tuple[List[str], List[np.ndarray], List[int]]:
    """1つのfoldのデータを読み込む
    
    Returns:
        (sample_ids, sequences, class_ids)
        - sequences: リスト of (T, D) numpy arrays（可変長）
    """
    df = pd.read_csv(csv_path, usecols=["file_path", "behavior"])
    
    sample_ids = []
    sequences = []
    class_ids = []
    
    failed_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {csv_path.name}", leave=False):
        file_path = str(row["file_path"])
        class_id = int(row["behavior"])
        
        # 画像を読み込み
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            failed_files.append(file_path)
            continue
        
        # 転置して (T, D) 形式に
        sequence = img.T
        sample_id = Path(file_path).stem
        
        sample_ids.append(sample_id)
        sequences.append(sequence)
        class_ids.append(class_id)
    
    if failed_files:
        print(f"[WARN] Failed to read {len(failed_files)} images in {csv_path.name}")
        for f in failed_files[:5]:  # 最初の5件のみ表示
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    return sample_ids, sequences, class_ids


def save_fold_as_npz(
    output_dir: Path,
    fold_id: str,
    sample_ids: List[str],
    sequences: List[np.ndarray],
    class_ids: List[int],
) -> Dict:
    """1つのfoldをNPZ形式（ragged array）で保存
    
    各サンプルを個別の配列として保存し、圧縮を適用。
    
    Returns:
        保存に関する統計情報の辞書
    """
    npz_path = output_dir / f"fold_{fold_id}.npz"
    
    # データ辞書を構築
    data_dict = {}
    for i, (sid, seq, cid) in enumerate(zip(sample_ids, sequences, class_ids)):
        # キー: sample_{i}_data, sample_{i}_id, sample_{i}_class
        data_dict[f"sample_{i}_data"] = seq
        data_dict[f"sample_{i}_id"] = sid
        data_dict[f"sample_{i}_class"] = cid
    
    # メタ情報を追加
    data_dict["num_samples"] = len(sample_ids)
    data_dict["fold_id"] = fold_id
    
    # 圧縮保存
    np.savez_compressed(npz_path, **data_dict)
    
    # 統計情報を収集
    file_size_mb = npz_path.stat().st_size / (1024 ** 2)
    shapes = [seq.shape for seq in sequences]
    time_lengths = [s[0] for s in shapes]
    feature_dims = [s[1] for s in shapes]
    
    stats = {
        "fold_id": fold_id,
        "num_samples": len(sample_ids),
        "file_size_mb": file_size_mb,
        "min_time_length": min(time_lengths),
        "max_time_length": max(time_lengths),
        "mean_time_length": np.mean(time_lengths),
        "feature_dim": feature_dims[0] if len(set(feature_dims)) == 1 else "mixed",
    }
    
    return stats


def save_metadata(
    output_dir: Path,
    fold_metadata: Dict[str, Dict[str, int]],
    conversion_stats: List[Dict],
) -> None:
    """メタデータをJSON形式で保存
    
    Args:
        fold_metadata: {fold_id: {sample_id: class_id}}
        conversion_stats: 各foldの変換統計情報リスト
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "format": "npz_ragged",
        "description": "10-fold ESN training data in NPY format (ragged arrays)",
        "folds": fold_metadata,
        "conversion_stats": conversion_stats,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Saved metadata to {metadata_path}")


def print_summary(conversion_stats: List[Dict]) -> None:
    """変換結果のサマリーを表示"""
    total_samples = sum(s["num_samples"] for s in conversion_stats)
    total_size_mb = sum(s["file_size_mb"] for s in conversion_stats)
    
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total folds:   {len(conversion_stats)}")
    print(f"Total samples: {total_samples}")
    print(f"Total size:    {total_size_mb:.2f} MB")
    print(f"Avg per fold:  {total_samples / len(conversion_stats):.1f} samples, {total_size_mb / len(conversion_stats):.2f} MB")
    print()
    
    print("Per-fold statistics:")
    print(f"{'Fold':<6} {'Samples':<8} {'Size(MB)':<10} {'T_min':<8} {'T_max':<8} {'T_mean':<8}")
    print("-" * 60)
    for stat in conversion_stats:
        print(
            f"{stat['fold_id']:<6} "
            f"{stat['num_samples']:<8} "
            f"{stat['file_size_mb']:<10.2f} "
            f"{stat['min_time_length']:<8} "
            f"{stat['max_time_length']:<8} "
            f"{stat['mean_time_length']:<8.1f}"
        )
    print("=" * 60)


def main():
    args = parse_args()
    
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    print(f"[INFO] CSV directory: {csv_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Format: {args.format}")
    
    # 入力検証
    available_folds = validate_inputs(csv_dir)
    
    # 出力ディレクトリのチェック
    if output_dir.exists() and not args.force and not args.dry_run:
        print(f"[ERROR] Output directory already exists: {output_dir}")
        print("        Use --force to overwrite or choose a different directory")
        sys.exit(1)
    
    if args.dry_run:
        print("[INFO] Dry run mode - no files will be written")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各foldを変換
    fold_metadata = {}
    conversion_stats = []
    
    print(f"\n[INFO] Starting conversion of {len(available_folds)} folds...")
    
    for fold_id in tqdm(available_folds, desc="Converting folds"):
        csv_path = csv_dir / f"10fold_{fold_id}.csv"
        
        # データ読み込み
        sample_ids, sequences, class_ids = load_fold_data(csv_path)
        
        # メタデータを記録
        fold_metadata[f"fold_{fold_id}"] = {
            sid: cid for sid, cid in zip(sample_ids, class_ids)
        }
        
        if not args.dry_run:
            # NPZ形式で保存
            stats = save_fold_as_npz(output_dir, fold_id, sample_ids, sequences, class_ids)
            conversion_stats.append(stats)
            print(f"[INFO] Saved fold_{fold_id}.npz ({stats['num_samples']} samples, {stats['file_size_mb']:.2f} MB)")
        else:
            print(f"[DRY RUN] Would save fold_{fold_id}.npz ({len(sample_ids)} samples)")
    
    # メタデータ保存
    if not args.dry_run:
        save_metadata(output_dir, fold_metadata, conversion_stats)
        print_summary(conversion_stats)
        print(f"\n[SUCCESS] Conversion complete! Output saved to {output_dir}")
    else:
        print("\n[DRY RUN] Validation complete. No files were written.")


if __name__ == "__main__":
    main()
