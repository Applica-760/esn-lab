"""
DataLoader Factory

設定に基づいて適切なDataLoaderインスタンスを生成するファクトリ関数。
"""
from pathlib import Path
from typing import Union

from .base import BaseDataLoader
from .csv_loader import CSVDataLoader
from .npy_loader import NPYDataLoader


def create_data_loader(
    data_source_cfg: dict,
    fallback_csv_dir: Union[str, Path, None] = None,
) -> BaseDataLoader:
    """設定に基づいてDataLoaderを生成
    
    Args:
        data_source_cfg: データソース設定辞書
            - type: "csv" | "npy"
            - csv_dir: CSVディレクトリ（type="csv"の場合）
            - npy_dir: NPYディレクトリ（type="npy"の場合）
        fallback_csv_dir: 後方互換性のためのCSVディレクトリ（data_source_cfg=Noneの場合）
    
    Returns:
        BaseDataLoader: CSVDataLoader または NPYDataLoader
    
    Raises:
        ValueError: 設定が不正な場合
        FileNotFoundError: 指定されたディレクトリが存在しない場合
    
    Examples:
        # 新方式（推奨）
        loader = create_data_loader({
            "type": "npy",
            "npy_dir": "dataset/10fold_npy/"
        })
        
        # 旧方式（後方互換）
        loader = create_data_loader(None, fallback_csv_dir="dataset/10fold_csvs/")
    """
    # 後方互換性: data_source_cfg が None の場合
    if data_source_cfg is None:
        if fallback_csv_dir is None:
            raise ValueError(
                "Either data_source_cfg or fallback_csv_dir must be provided"
            )
        print("[INFO] Using CSV data loader (legacy mode)")
        return CSVDataLoader(fallback_csv_dir)
    
    # data_source_cfg から type を取得
    data_type = data_source_cfg.get("type", "csv")
    
    if data_type == "csv":
        csv_dir = data_source_cfg.get("csv_dir")
        if csv_dir is None:
            # fallback_csv_dir を使用
            if fallback_csv_dir is None:
                raise ValueError(
                    "data_source.type='csv' but csv_dir is not specified"
                )
            csv_dir = fallback_csv_dir
        
        print(f"[INFO] Using CSV data loader: {csv_dir}")
        return CSVDataLoader(csv_dir)
    
    elif data_type == "npy":
        npy_dir = data_source_cfg.get("npy_dir")
        if npy_dir is None:
            raise ValueError(
                "data_source.type='npy' but npy_dir is not specified"
            )
        
        print(f"[INFO] Using NPY data loader: {npy_dir}")
        return NPYDataLoader(npy_dir)
    
    else:
        raise ValueError(
            f"Unknown data_source.type: '{data_type}'. "
            "Supported types are: 'csv', 'npy'"
        )


def create_data_loader_from_config(cfg, tenfold_cfg=None) -> tuple[BaseDataLoader, Path]:
    """Config オブジェクトからDataLoaderとデータディレクトリを生成
    
    Args:
        cfg: Config オブジェクト
        tenfold_cfg: TrainTenfoldCfg または同等のオブジェクト（Noneの場合 cfg.train.tenfold を使用）
    
    Returns:
        tuple[BaseDataLoader, Path]: (data_loader, data_dir)
            - data_loader: 生成されたデータローダー
            - data_dir: データディレクトリ（csv_dir または npy_dir）
    """
    tenfold_cfg = tenfold_cfg or getattr(getattr(cfg, "train", None), "tenfold", None)
    if tenfold_cfg is None:
        raise ValueError("Config 'cfg.train.tenfold' not found.")
    
    # data_source が設定されているか確認
    data_source = getattr(tenfold_cfg, "data_source", None)
    csv_dir_fallback = getattr(tenfold_cfg, "csv_dir", None)
    
    if data_source is not None:
        # 新方式: data_source を使用
        data_source_dict = {
            "type": getattr(data_source, "type", "csv"),
            "csv_dir": getattr(data_source, "csv_dir", None),
            "npy_dir": getattr(data_source, "npy_dir", None),
        }
        loader = create_data_loader(data_source_dict, fallback_csv_dir=csv_dir_fallback)
        
        # data_dir を決定
        if data_source_dict["type"] == "npy":
            data_dir = Path(data_source_dict["npy_dir"]).expanduser().resolve()
        else:
            csv_dir = data_source_dict["csv_dir"] or csv_dir_fallback
            data_dir = Path(csv_dir).expanduser().resolve()
    
    else:
        # 旧方式: csv_dir のみ使用（後方互換）
        if csv_dir_fallback is None:
            raise ValueError(
                "Neither data_source nor csv_dir is specified in tenfold config"
            )
        loader = create_data_loader(None, fallback_csv_dir=csv_dir_fallback)
        data_dir = Path(csv_dir_fallback).expanduser().resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    return loader, data_dir
