# pyesn/setup/registry.py

# scheme
from .config import (TrainSingleCfg, TrainBatchCfg, TrainTenfoldCfg,
                     PredictSingleCfg, PredictBatchCfg,
                     EvaluateRunCfg)
# runner
from ..runner.train import single_train, batch_train
from ..runner.tenfold.main import run_tenfold
from ..runner.predict import single_predict, batch_predict
from ..runner.evaluate import single_evaluate


REGISTRY = {
    "train": {
        "variants": {
            "single": {"schema": TrainSingleCfg, "runner": single_train},
            "batch":  {"schema": TrainBatchCfg,  "runner": batch_train},
            "tenfold": {"schema": TrainTenfoldCfg, "runner": run_tenfold},
        }
    },
    "predict": {
        "variants": {
            "single": {"schema": PredictSingleCfg, "runner": single_predict},
            "batch":  {"schema": PredictBatchCfg, "runner": batch_predict}
        }
    },
    "evaluate": {
        "variants": {
            "run": {"schema": EvaluateRunCfg, "runner": single_evaluate},
            "cv" : {}
        }
    },
    "integ": {
        "variants": {}
    },
}