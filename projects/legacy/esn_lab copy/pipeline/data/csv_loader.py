"""
CSV+画像ファイル形式のデータローダー

既存のCSV形式（10fold_*.csv）と画像ファイルから
時系列データを読み込むローダー実装。
"""
from pathlib import Path
from typing import Iterator
import pandas as pd
import cv2
import numpy as np

from .base import BaseDataLoader


class CSVDataLoader(BaseDataLoader):
    """CSV+画像ファイルからデータを読み込むローダー
    
    10-fold交差検証用のCSVファイル（10fold_a.csv, 10fold_b.csv, ...）から
    画像パスとラベルを読み込み、実行時に画像をロードする。
    
    Args:
        csv_dir: CSVファイルが格納されているディレクトリ
    """

    def __init__(self, csv_dir: str | Path):
        self.csv_dir = Path(csv_dir)
        if not self.csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {self.csv_dir}")
        
        # 利用可能なfoldを検出
        self._available_folds = self._detect_available_folds()
        if not self._available_folds:
            raise ValueError(f"No 10fold_*.csv files found in {self.csv_dir}")

    def _detect_available_folds(self) -> list[str]:
        """利用可能なfold IDを検出"""
        folds = []
        for ch in [chr(c) for c in range(ord("a"), ord("j") + 1)]:
            csv_path = self.csv_dir / f"10fold_{ch}.csv"
            if csv_path.exists():
                folds.append(ch)
        return sorted(folds)

    def get_available_folds(self) -> list[str]:
        """利用可能なfold ID一覧を返す"""
        return self._available_folds.copy()

    def get_fold_size(self, fold_id: str) -> int:
        """指定されたfoldのサンプル数を取得"""
        csv_path = self.csv_dir / f"10fold_{fold_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path, usecols=["file_path"])
        return len(df)

    def load_fold_data(
        self,
        fold_ids: list[str],
    ) -> Iterator[tuple[str, np.ndarray, int]]:
        """指定されたfoldのデータをイテレータで返す
        
        Args:
            fold_ids: 読み込むfoldのID一覧（例: ['a', 'b', 'c']）
        
        Yields:
            tuple[str, np.ndarray, int]: (sample_id, sequence, class_id)
        """
        # 各foldのCSVを順に処理
        for fold_id in fold_ids:
            csv_path = self.csv_dir / f"10fold_{fold_id}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # CSVを読み込み
            usecols = ["file_path", "behavior"]
            df = pd.read_csv(csv_path, usecols=usecols)
            
            if list(df.columns) != usecols:
                raise ValueError(
                    f"CSV columns mismatch at {csv_path}. "
                    f"expected={usecols}, got={list(df.columns)}"
                )
            
            # 各サンプルを順に処理
            for _, row in df.iterrows():
                file_path = str(row["file_path"])
                class_id = int(row["behavior"])
                
                # 画像を読み込み
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {file_path}")
                
                # 転置して (T, D) 形式に変換
                sequence = img.T
                
                # サンプルIDを抽出
                sample_id = Path(file_path).stem
                
                yield sample_id, sequence, class_id
