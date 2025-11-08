from pathlib import Path
import numpy as np

from esn_lab.pipeline.eval.evaluator import Evaluator
from esn_lab.pipeline.pred.predictor import Predictor
from esn_lab.pipeline.data import BaseDataLoader, create_data_loader_from_config
from esn_lab.model.model_builder import get_model


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
        data_loader: BaseDataLoader,
        overrides: dict,
        train_tag: str,
        holdout: str,
    ) -> tuple[dict, list[dict]]:
        # Build model and load weight
        model, _ = get_model(cfg, overrides)
        weight = np.load(weight_path, allow_pickle=True)
        model.Output.setweight(weight)

        # Load holdout data using the provided data_loader
        # データをリストに収集
        ids = []
        sequences = []
        class_ids = []
        for sample_id, U, class_id in data_loader.load_fold_data([holdout]):
            ids.append(sample_id)
            sequences.append(U)
            class_ids.append(class_id)

        row, pred_rows = self._evaluator.evaluate_dataset_majority(
            cfg=cfg,
            model=model,
            predictor=self._predictor,
            ids=ids,
            sequences=sequences,
            class_ids=class_ids,
            wf_name=Path(weight_path).name,
            train_tag=train_tag,
            holdout=holdout,
            overrides=overrides,
        )
        return row, pred_rows


def eval_one_weight_worker(cfg, weight_path: str, ten_cfg, overrides: dict, train_tag: str, holdout: str):
    """Executor向けワーカ関数。プロセス側でTenfoldEvaluatorを生成し評価を行う。

    data_loaderはspawnモードでpickleできないため、ワーカー内で再作成する。
    """
    # ワーカープロセス内でdata_loaderを再作成
    data_loader, _ = create_data_loader_from_config(cfg, ten_cfg)

    evaluator = TenfoldEvaluator(cfg.run_dir)
    return evaluator.eval_weight_on_holdout(
        cfg=cfg,
        weight_path=Path(weight_path),
        data_loader=data_loader,
        overrides=overrides,
        train_tag=train_tag,
        holdout=holdout,
    )
