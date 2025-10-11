__version__ = "0.1.0"

# --- configs ---
from .utils.config import (
    Config,
    TrainSingleCfg,
    TrainBatchCfg,
    TrainTenfoldCfg,
    TrainTenfoldSearchCfg,
    PredictSingleCfg,
    PredictBatchCfg,
    EvaluateRunCfg,
)

# --- runners ---
from .runner.train import single_train, batch_train
from .runner.train_10fold import tenfold_search_train
from .runner.predict import single_predict, batch_predict
from .runner.evaluate import single_evaluate

# --- core models/pipelines ---
from .model.esn import ESN
from .pipeline.trainer import Trainer
from .pipeline.predictor import Predictor
from .pipeline.evaluator import Evaluator

from .model.model_builder import get_model