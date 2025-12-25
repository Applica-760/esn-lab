"""
NPY形式のデータローダー（前処理済みバイナリ用）

前処理スクリプト（convert_10fold_to_npy.py）で生成された
NPZ形式のデータを高速に読み込むローダー実装。
"""
from pathlib import Path
from typing import Iterator
import json
import numpy as np

from .base import BaseDataLoader


class NPYDataLoader(BaseDataLoader):
    """NPZ形式（Ragged array）からデータを読み込むローダー
    
    前処理済みのNPZファイルから時系列データを読み込む。
    各foldは fold_{id}.npz として保存されており、内部には:
    - sample_{i}_data: (T_i, D) の時系列配列
    - sample_{i}_id: サンプルID文字列
    - sample_{i}_class: クラスID整数
    - num_samples: サンプル数
    - fold_id: fold ID
    
    Args:
        npy_dir: NPZファイルとmetadata.jsonが格納されているディレクトリ
    """

    def __init__(self, npy_dir: str | Path):
        self.npy_dir = Path(npy_dir)
        if not self.npy_dir.exists():
            raise FileNotFoundError(f"NPY directory not found: {self.npy_dir}")
        
        # メタデータを読み込み
        metadata_path = self.npy_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.npy_dir}. "
                "Please run convert_10fold_to_npy.py first."
            )
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        # 利用可能なfoldを検出
        self._available_folds = self._detect_available_folds()
        if not self._available_folds:
            raise ValueError(f"No fold_*.npz files found in {self.npy_dir}")

    def _detect_available_folds(self) -> list[str]:
        """利用可能なfold IDを検出"""
        folds = []
        for ch in [chr(c) for c in range(ord("a"), ord("j") + 1)]:
            npz_path = self.npy_dir / f"fold_{ch}.npz"
            if npz_path.exists():
                folds.append(ch)
        return sorted(folds)

    def get_available_folds(self) -> list[str]:
        """利用可能なfold ID一覧を返す"""
        return self._available_folds.copy()

    def get_fold_size(self, fold_id: str) -> int:
        """指定されたfoldのサンプル数を取得"""
        npz_path = self.npy_dir / f"fold_{fold_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        # メタデータから取得（ファイルを開かずに）
        fold_key = f"fold_{fold_id}"
        if fold_key in self.metadata.get("folds", {}):
            return len(self.metadata["folds"][fold_key])
        
        # メタデータにない場合はファイルを開いて確認
        with np.load(npz_path, allow_pickle=True) as data:
            return int(data["num_samples"])

    def load_fold_data(
        self,
        fold_ids: list[str],
    ) -> Iterator[tuple[str, np.ndarray, int]]:
        """指定されたfoldのデータをイテレータで返す
        
        Args:
            fold_ids: 読み込むfoldのID一覧（例: ['a', 'b', 'c']）
        
        Yields:
            tuple[str, np.ndarray, int]: (sample_id, sequence, class_id)
                - sample_id: サンプルの識別子
                - sequence: 時系列データ (T, D) の2次元配列
                - class_id: クラスラベル（整数）
        
        Notes:
            - mmap_modeを使用せず、明示的にcopy=Trueでコピーを作成します
            - これによりメモリマップの寿命問題を回避し、安全性を確保します
        """
        for fold_id in fold_ids:
            npz_path = self.npy_dir / f"fold_{fold_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ file not found: {npz_path}")
            
            # NPZファイルを読み込み（mmap_modeは使用しない）
            with np.load(npz_path, allow_pickle=True) as data:
                num_samples = int(data["num_samples"])
                
                # 各サンプルを順に yield（明示的にコピーを作成）
                for i in range(num_samples):
                    sample_id = str(data[f"sample_{i}_id"])
                    # copy=True を明示して、withブロック外でも安全に使える配列を作成
                    sequence = np.array(data[f"sample_{i}_data"], copy=True)
                    class_id = int(data[f"sample_{i}_class"])
                    
                    yield sample_id, sequence, class_id
