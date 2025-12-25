#!/usr/bin/env python3
"""
10fold_npy_divディレクトリから各npzファイルのid/classを抽出し、CSV形式で出力するスクリプト

python tools/data_analysis/make_divdata_table.py \
    --input_dir dataset/10fold_npy_div \
    --output_dir dataset/divdata_tables
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple


def load_npz_and_extract(npz_path: Path) -> pd.DataFrame:
    """npzファイルからid/classを抽出してDataFrameを作成
    
    Args:
        npz_path: npzファイルのパス
    
    Returns:
        id/classカラムを持つDataFrame
    
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        KeyError: 必要なキーが存在しない場合
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"npzファイルが見つかりません: {npz_path}")
    
    data = np.load(npz_path)
    
    # num_samplesを取得
    if 'num_samples' not in data:
        raise KeyError(f"'num_samples'キーが見つかりません: {npz_path}")
    
    num_samples = int(data['num_samples'])
    
    # 各サンプルのid/classを抽出
    records = []
    for i in range(num_samples):
        id_key = f'sample_{i}_id'
        class_key = f'sample_{i}_class'
        
        if id_key not in data or class_key not in data:
            raise KeyError(f"サンプル{i}のキーが見つかりません: {npz_path}")
        
        sample_id = str(data[id_key])
        sample_class = int(data[class_key])
        records.append({'id': sample_id, 'class': sample_class})
    
    return pd.DataFrame(records)


def process_fold_directory(input_subdir: Path, output_subdir: Path) -> None:
    """1つのfoldディレクトリ（a~j）を処理
    
    Args:
        input_subdir: 入力サブディレクトリ（例: dataset/10fold_npy_div/a）
        output_subdir: 出力サブディレクトリ（例: outputs/divdata_tables/a）
    
    Raises:
        FileNotFoundError: npzファイルが見つからない場合
    """
    # 出力ディレクトリを作成
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # すべての.npzファイルを取得
    npz_files = sorted(input_subdir.glob("fold_*.npz"))
    
    if not npz_files:
        raise FileNotFoundError(f"npzファイルが見つかりません: {input_subdir}")
    
    print(f"\n処理中: {input_subdir.name}/")
    for npz_file in npz_files:
        # CSVファイル名を生成
        csv_file = output_subdir / f"{npz_file.stem}.csv"
        
        # データを抽出
        df = load_npz_and_extract(npz_file)
        
        # CSVとして保存（インデックスなし、ヘッダーあり）
        df.to_csv(csv_file, index=False)
        
        print(f"  {npz_file.name} -> {csv_file.name} ({len(df)} records)")


def main():
    parser = argparse.ArgumentParser(
        description="10fold_npy_divディレクトリからid/classテーブルを生成"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="入力ディレクトリ（例: dataset/10fold_npy_div）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="出力ディレクトリ（例: outputs/divdata_tables）"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 入力ディレクトリの存在確認
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    
    print(f"入力: {input_dir}")
    print(f"出力: {output_dir}")
    
    # a~jのサブディレクトリを処理
    fold_chars = 'abcdefghij'
    processed_count = 0
    
    for fold_char in fold_chars:
        input_subdir = input_dir / fold_char
        
        if not input_subdir.exists():
            print(f"\nスキップ: {fold_char}/ (ディレクトリが存在しません)")
            continue
        
        output_subdir = output_dir / fold_char
        process_fold_directory(input_subdir, output_subdir)
        processed_count += 1
    
    print(f"\n完了: {processed_count}個のディレクトリを処理しました")


if __name__ == "__main__":
    main()
