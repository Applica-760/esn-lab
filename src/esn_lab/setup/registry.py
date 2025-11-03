# esn_lab/setup/registry.py

# scheme
from .config import (TrainSingleCfg, TrainBatchCfg, TrainTenfoldCfg,
                     PredictSingleCfg, PredictBatchCfg,
                     EvaluateRunCfg, EvaluateTenfoldCfg, EvaluateSummaryCfg, EvaluateAnalysisCfg)
# runner
from ..runner.train.train import single_train, batch_train
from ..runner.train.tenfold.main import run_tenfold
from ..runner.pred.predict import single_predict, batch_predict
from ..runner.eval.evaluate import single_evaluate, tenfold_evaluate, summary_evaluate
from ..runner.eval.analysis import analysis_evaluate
from ..runner.integ.grid import run_grid
from .config import IntegGridCfg


REGISTRY = {
    "train": {
        "variants": {
            "single": {"schema": TrainSingleCfg, "runner": single_train},
            "batch":  {"schema": TrainBatchCfg,  "runner": batch_train},
            "tenfold": {"schema": TrainTenfoldCfg, "runner": run_tenfold},
        }
    },
    "pred": {
        "variants": {
            "single": {"schema": PredictSingleCfg, "runner": single_predict},
            "batch":  {"schema": PredictBatchCfg, "runner": batch_predict}
        }
    },
    "eval": {
        "variants": {
            "run": {"schema": EvaluateRunCfg, "runner": single_evaluate},
            "cv" : {},
            "tenfold": {"schema": EvaluateTenfoldCfg, "runner": tenfold_evaluate},
            "summary": {"schema": EvaluateSummaryCfg, "runner": summary_evaluate},
            "analysis": {"schema": EvaluateAnalysisCfg, "runner": analysis_evaluate},
        }
    },
    "integ": {
        "variants": {
            "grid": {"schema": IntegGridCfg, "runner": run_grid}
        }
    },
}