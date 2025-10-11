# runner/evaluate.py
from pathlib import Path

from pyesn.setup.config import Config
from pyesn.pipeline.evaluator import Evaluator
from pyesn.utils.io import load_jsonl, target_output_from_dict
from pyesn.utils.constants import PREDICT_RECORD_FILE

def single_evaluate(cfg: Config):
    run_dir = Path(cfg.evaluate.run.run_dir)
    
    file = list(run_dir.glob(PREDICT_RECORD_FILE))[0]
    datas = load_jsonl(file)

    evaluator = Evaluator()

    # ↓正解ラベル(時間あたりでなくデータ単位)と予測ラベルのペアをループで取得

    for i, data in enumerate(datas):
        record = target_output_from_dict(data)
        # evaluator.evaluate_classification_result(record)
        evaluator.majority_success(record)

    # ↑で取得したペアのデータを使って，混同行列を作る

    return
