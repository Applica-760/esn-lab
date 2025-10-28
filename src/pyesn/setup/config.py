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
    csv_dir: Optional[str] = None
    workers: Optional[int] = None
    # 統一名: weight_dir のみを使用
    weight_dir: Optional[str] = None
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

@dataclass
class PredictBatchCfg:
    ids: Optional[list[str]] = None
    paths: Optional[list[str]] = None
    class_ids: Optional[list[int]] = None
    weight: Optional[str] = None

@dataclass
class Predict:
    single: Optional[PredictSingleCfg] = None
    batch: Optional[PredictBatchCfg] = None



# evaluate ==============================================================
@dataclass
class EvaluateTenfoldCfg:
    csv_dir: Optional[str] = None
    weight_dir: Optional[str] = None
    workers: Optional[int] = None 
    parallel: Optional[bool] = True

@dataclass
class EvaluateRunCfg:
    run_dir: Optional[str] = None

@dataclass
class EvaluateSummaryCfg:
    weight_dir: Optional[str] = None
    csv_name: Optional[str] = "evaluation_results.csv"
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
class Evaluate:
    run: Optional[EvaluateRunCfg] = None
    tenfold: Optional[EvaluateTenfoldCfg] = None
    summary: Optional[EvaluateSummaryCfg] = None



# integ ==============================================================
@dataclass
class IntegGridCfg:
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
