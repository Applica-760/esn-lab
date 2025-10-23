# utils/config.py
from typing import Optional
from dataclasses import dataclass


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
    weight_path: Optional[str] = None
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

@dataclass
class EvaluateRunCfg:
    run_dir: Optional[str] = None

@dataclass
class EvaluateSummaryCfg:
    # Path to directory that contains evaluation_results.csv
    weight_dir: Optional[str] = None
    # CSV file name inside weight_dir
    csv_name: Optional[str] = "evaluation_results.csv"
    # Which metric to visualize: "majority_acc" or "timestep_acc"
    metric: Optional[str] = "majority_acc"
    # Vary one hyperparameter and plot mean±std as error bars over its values
    vary_param: Optional[str] = "Nx"
    vary_values: Optional[list] = None  # e.g., [200, 300, 400]
    # Filters to apply before pivot. Example: {"Nx": 200, "density": 0.5}
    filters: Optional[dict] = None
    # Aggregation function when multiple points map to same grid cell
    agg: Optional[str] = "mean"
    # Output directory (default: weight_dir/evaluation_plots)
    output_dir: Optional[str] = None
    # Plot options
    fmt: Optional[str] = ".3f"  # number formatting for CSV/logs
    title: Optional[str] = None
    dpi: Optional[int] = 150

@dataclass
class Evaluate:
    run: Optional[EvaluateRunCfg] = None
    tenfold: Optional[EvaluateTenfoldCfg] = None
    summary: Optional[EvaluateSummaryCfg] = None



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
