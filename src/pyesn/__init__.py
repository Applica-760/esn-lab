__version__ = "0.1.0"

# --- configs ---
from .setup.config import (
    Config,
    TrainSingleCfg,
    TrainBatchCfg,
    TrainTenfoldCfg,
    PredictSingleCfg,
    PredictBatchCfg,
    EvaluateRunCfg,
)

# --- runners ---
from .runner.train.train import single_train, batch_train
from .runner.train.tenfold.main import run_tenfold
from .runner.pred.predict import single_predict, batch_predict
from .runner.eval.evaluate import single_evaluate

# --- core models/pipelines ---
from .model.esn import ESN
from .pipeline.train.trainer import Trainer
from .pipeline.pred.predictor import Predictor
from .pipeline.eval.evaluator import Evaluator

from .model.model_builder import get_model