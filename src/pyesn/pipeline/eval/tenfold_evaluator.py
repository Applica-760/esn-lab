from pathlib import Path
import numpy as np

from pyesn.pipeline.eval.evaluator import Evaluator
from pyesn.pipeline.pred.predictor import Predictor
from pyesn.pipeline.tenfold_util import load_10fold_csv_mapping, read_data_from_csvs
from pyesn.model.model_builder import get_model


class TenfoldEvaluator:
    """tenfold評価の処理本体（pipeline側）。

    - 重みファイルの読み込み、ホールドアウトfoldデータの読み込み、推論/評価を担当。
    - 並列制御や結果CSVの集約はrunner側の責務。
    """

    def __init__(self, run_dir: str):
        self._predictor = Predictor(run_dir)
        self._evaluator = Evaluator()

    def eval_weight_on_holdout(
        self,
        cfg,
        weight_path: Path,
        csv_dir: Path,
        overrides: dict,
        train_tag: str,
        holdout: str,
    ) -> tuple[dict, list[dict]]:
        # Build model and load weight
        model, _ = get_model(cfg, overrides)
        weight = np.load(weight_path, allow_pickle=True)
        model.Output.setweight(weight)

        # Load holdout data
        csv_map = load_10fold_csv_mapping(Path(csv_dir))
        ids, paths, class_ids = read_data_from_csvs([csv_map[holdout]])
        assert len(ids) == len(paths) == len(class_ids), "length mismatch"

        row, pred_rows = self._evaluator.evaluate_dataset_majority(
            cfg=cfg,
            model=model,
            predictor=self._predictor,
            ids=ids,
            paths=paths,
            class_ids=class_ids,
            wf_name=Path(weight_path).name,
            train_tag=train_tag,
            holdout=holdout,
            overrides=overrides,
        )
        return row, pred_rows


def eval_one_weight_worker(cfg, weight_path: str, csv_dir: str, overrides: dict, train_tag: str, holdout: str):
    """Executor向けワーカ関数。プロセス側でTenfoldEvaluatorを生成し評価を行う。"""
    evaluator = TenfoldEvaluator(cfg.run_dir)
    return evaluator.eval_weight_on_holdout(
        cfg=cfg,
        weight_path=Path(weight_path),
        csv_dir=Path(csv_dir),
        overrides=overrides,
        train_tag=train_tag,
        holdout=holdout,
    )
