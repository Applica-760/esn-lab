import cv2
import time
from datetime import datetime
from pathlib import Path

from esn_lab.pipeline.train.trainer import Trainer
from esn_lab.pipeline.tenfold_util import read_data_from_csvs
from esn_lab.pipeline.tenfold_util import make_weight_filename
from esn_lab.utils.data_processing import make_onehot
from esn_lab.model.model_builder import get_model


class TenfoldTrainer:
    """10-fold 学習の処理本体（pipeline側）。

    - データ読み込み、モデル構築、学習、重み保存までを担当。
    - 乱数シード設定や並列実行の制御はrunner側の責務とする。
    """

    def __init__(self, run_dir: str):
        self._trainer = Trainer(run_dir)

    def run_one_fold_search(
        self,
        cfg,
        csv_map: dict[str, Path],
        all_letters: list[str],
        leave_out_letter: str,
        hp_overrides: dict,
        weight_dir: Path,
    ) -> tuple[float, str]:
        """1-foldの学習を実行し、(実行時間, 完了タイムスタンプ)を返す。"""
        start_time = time.monotonic()

        train_letters = [x for x in all_letters if x != leave_out_letter]
        train_tag = "".join(train_letters)
        print(f"[INFO] Start 10-fold train (leave_out='{leave_out_letter}') for hyperparams: {hp_overrides or 'default'}")

        csv_paths = [csv_map[ch] for ch in train_letters]
        ids, paths, class_ids = read_data_from_csvs(csv_paths)
        assert len(ids) == len(paths) == len(class_ids), "length mismatch"

        model, optimizer = get_model(cfg, hp_overrides)

        Ny = cfg.model.Ny
        for i in range(len(paths)):
            img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {paths[i]}")
            U = img.T
            T = len(U)
            D = make_onehot(class_ids[i], T, Ny)
            self._trainer.train(model, optimizer, ids[i], U, D)

        weight_file_name = make_weight_filename(cfg=cfg, overrides=hp_overrides, train_tag=train_tag)
        self._trainer.save_output_weight(
            Wout=model.Output.Wout,
            file_name=weight_file_name,
            save_dir=str(weight_dir)
        )
        print(f"[INFO] Finished fold '{leave_out_letter}'. Weight saved to {weight_dir / weight_file_name}")

        end_time = time.monotonic()
        execution_time = end_time - start_time
        timestamp = datetime.now().isoformat()

        return execution_time, timestamp
