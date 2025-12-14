# utils/config.py
from typing import Optional
from dataclasses import dataclass, field


# util ==============================================================
@dataclass
class Empty: pass

@dataclass
class TargetOutputData:
    target_series: list
    output_series: list

@dataclass
class TargetOutput:
    id: Optional[str] = None
    data: Optional[TargetOutputData] = None

# data ==============================================================
@dataclass
class DataSourceCfg:
    """データソース設定
    
    type: "csv" | "npy"
        - "csv": CSV + 画像ファイル形式（既存）
        - "npy": 前処理済みNPY/NPZ形式
    csv_dir: CSVディレクトリ（type="csv"の場合）
    npy_dir: NPYディレクトリ（type="npy"の場合）
    """
    type: str = "csv"  # "csv" or "npy"
    csv_dir: Optional[str] = None
    npy_dir: Optional[str] = None

# model ==============================================================
@dataclass
class ModelCfg:
    name: str = "esn"
    Nu: int = 256
    Nx: int = 10
    Ny: int = 3
    density: float = 0.1
    input_scale: float = 0.01
    rho: float = 0.9
    optimizer: str = "tikhonov"




# train ==============================================================
@dataclass
class TrainSingleCfg:
    id: Optional[str] = None
    path: Optional[str] = None
    class_id: Optional[int] = None

@dataclass
class TrainBatchCfg:
    ids: Optional[list[str]] = None
    paths: Optional[list[str]] = None
    class_ids: Optional[list[int]] = None

@dataclass
class TrainTenfoldCfg:
    # データソース設定（新方式: 推奨）
    data_source: Optional[DataSourceCfg] = None
    # 後方互換性のため csv_dir も残す（data_source より優先度低）
    csv_dir: Optional[str] = None
    # 実験名を指定（outputs/experiments/{experiment_name}/ に配置）
    experiment_name: Optional[str] = None
    workers: Optional[int] = None
    # スキップ制御フラグ
    skip_existing: Optional[bool] = True
    force_retrain: Optional[bool] = False
    search_space: Optional[dict[str, list]] | None = None

@dataclass
class Train:
    single: Optional[TrainSingleCfg] = None
    batch: Optional[TrainBatchCfg] = None
    tenfold: Optional[TrainTenfoldCfg] = None


# predict ==============================================================
@dataclass
class PredictSingleCfg:
    id: Optional[str] = None
    path: Optional[str] = None
    class_id: Optional[int] = None
    weight: Optional[str] = None
    # データソース設定（新方式: 推奨）
    data_source: Optional[DataSourceCfg] = None

@dataclass
class PredictBatchCfg:
    ids: Optional[list[str]] = None
    paths: Optional[list[str]] = None
    class_ids: Optional[list[int]] = None
    weight: Optional[str] = None
    # データソース設定（新方式: 推奨）
    data_source: Optional[DataSourceCfg] = None

@dataclass
class Predict:
    single: Optional[PredictSingleCfg] = None
    batch: Optional[PredictBatchCfg] = None



# evaluate ==============================================================
@dataclass
class EvaluateTenfoldCfg:
    # データソース設定（新方式: 推奨）
    data_source: Optional[DataSourceCfg] = None
    # 後方互換性のため csv_dir も残す（data_source より優先度低）
    csv_dir: Optional[str] = None
    experiment_name: Optional[str] = None
    workers: Optional[int] = None 
    parallel: Optional[bool] = True
    # Optional: limit evaluation to specific parameter combinations.
    # Use the same schema as TrainTenfoldCfg.search_space: keys like 'model.Nx'.
    search_space: Optional[dict[str, list]] | None = None

@dataclass
class EvaluateRunCfg:
    run_dir: Optional[str] = None

@dataclass
class EvaluateSummaryCfg:
    experiment_name: Optional[str] = None
    # 明示的なCSVパス指定（experiment_nameより優先）
    results_csv: Optional[str] = None
    predictions_csv: Optional[str] = None
    metric: Optional[str] = "majority_acc"
    vary_param: Optional[str] = "Nx"
    vary_values: Optional[list] = None  
    filters: Optional[dict] = None
    agg: Optional[str] = "mean"
    output_dir: Optional[str] = None
    fmt: Optional[str] = ".3f" 
    title: Optional[str] = None
    dpi: Optional[int] = 150

@dataclass
class EvaluateAnalysisCfg:
    experiment_name: Optional[str] = None
    # 明示的なCSVパス指定（experiment_nameより優先）
    predictions_csv: Optional[str] = None
    filters: Optional[dict] = None
    output_dir: Optional[str] = None
    # Optional dataset CSV directory (10-fold) to enable locating and exporting images
    csv_dir: Optional[str] = None
    # Whether to export/copy extreme images into subfolders
    export_images: Optional[bool] = True
    title: Optional[str] = None
    dpi: Optional[int] = 150

@dataclass
class Evaluate:
    run: Optional[EvaluateRunCfg] = None
    tenfold: Optional[EvaluateTenfoldCfg] = None
    summary: Optional[EvaluateSummaryCfg] = None
    analysis: Optional[EvaluateAnalysisCfg] = None



# integ ==============================================================
@dataclass
class IntegGridCfg:
    # 新スキーマ: grid.yaml を単一ファイルで受け、
    # - base: csv_dir/weight_dir/workers などの共通デフォルト
    # - param_grid: {"model.<field>": [values, ...]} の探索範囲（flatten_search_space が解釈）
    experiment_name: Optional[str] = None
    base: Optional[dict] = None
    param_grid: Optional[dict[str, list]] = None
    train: Optional[TrainTenfoldCfg] = None
    # eval は Evaluate の tenfold/summary を再利用（run は未使用）
    eval: Optional[Evaluate] = None

@dataclass
class Integ:
    grid: Optional[IntegGridCfg] = None

@dataclass
class Config:
    project: str
    seeds: list[int]
    num_of_classes: int
    data: dict
    run_dir: str
    model: ModelCfg = field(default_factory=ModelCfg)
    train: Optional[Train] = field(default_factory=Train)
    predict: Optional[Predict] = field(default_factory=Predict)
    evaluate: Optional[Evaluate] = field(default_factory=Evaluate)
    integ: Optional[Integ] = field(default_factory=Integ)
