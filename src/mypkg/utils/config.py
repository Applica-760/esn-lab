# utils/config.py
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class TrainTenfoldSearchCfg:
    csv_dir: Optional[str] = None
    workers: Optional[int] = None
    weight_path: Optional[str] = None
    search_space: Optional[dict[str, list]] | None = None

@dataclass
class Train:
    single: Optional[TrainSingleCfg] = None
    batch: Optional[TrainBatchCfg] = None
    tenfold: Optional[TrainTenfoldCfg] = None
    tenfold_search: Optional[TrainTenfoldSearchCfg] = None


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
class EvaluateRunCfg:
    run_dir: Optional[str] = None

@dataclass
class Evaluate:
    run: Optional[EvaluateRunCfg] = None


# integ ==============================================================
@dataclass
class Config:
    project: str
    seeds: list[int]
    num_of_classes: int
    data: dict
    run_dir: str
    model: ModelCfg = ModelCfg()
    train: Optional[Train] = Train
    predict: Optional[Predict] = Predict
    evaluate: Optional[Evaluate] = Evaluate
