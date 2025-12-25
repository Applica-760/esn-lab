"""
データローダーの抽象基底クラス

ESNの訓練・評価で使用する時系列データを読み込むための
統一インターフェースを提供します。
"""
from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np


class BaseDataLoader(ABC):
    """時系列データローダーの抽象基底クラス
    
    異なるデータソース（CSV+画像、NPY、HDF5など）から
    統一された形式でデータを提供するためのインターフェース。
    """

    @abstractmethod
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
                    T: 時系列長（可変）
                    D: 特徴次元
                - class_id: クラスラベル（整数）
        
        Notes:
            - メモリ効率のためイテレータで返す（全データを一度に載せない）
            - 各サンプルの時系列長Tは異なってもよい（Ragged array対応）
        """
        pass

    @abstractmethod
    def get_available_folds(self) -> list[str]:
        """利用可能なfold IDの一覧を取得
        
        Returns:
            list[str]: fold ID一覧（例: ['a', 'b', 'c', ..., 'j']）
        """
        pass

    @abstractmethod
    def get_fold_size(self, fold_id: str) -> int:
        """指定されたfoldのサンプル数を取得
        
        Args:
            fold_id: fold ID
        
        Returns:
            int: サンプル数
        """
        pass
