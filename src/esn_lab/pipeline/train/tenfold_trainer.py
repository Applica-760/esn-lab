import time
from datetime import datetime
from pathlib import Path

from esn_lab.pipeline.train.trainer import Trainer
from esn_lab.pipeline.data import BaseDataLoader
from esn_lab.pipeline.tenfold_util import make_weight_filename
from esn_lab.utils.data_processing import make_onehot
from esn_lab.model.model_builder import get_model
from esn_lab.utils.io import save_numpy_npy_atomic


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
        data_loader: BaseDataLoader,
        all_letters: list[str],
        leave_out_letter: str,
        hp_overrides: dict,
        weight_dir: Path,
    ) -> tuple[float, str]:
        """1-foldの学習を実行し、(実行時間, 完了タイムスタンプ)を返す。
        
        Args:
            cfg: 設定オブジェクト
            data_loader: データローダーインスタンス
            all_letters: 全fold ID一覧
            leave_out_letter: テストfold ID（学習から除外）
            hp_overrides: ハイパーパラメータの上書き
            weight_dir: 重み保存先ディレクトリ
        
        Returns:
            tuple[float, str]: (実行時間[秒], 完了タイムスタンプ)
        """
        start_time = time.monotonic()

        train_letters = [x for x in all_letters if x != leave_out_letter]
        train_tag = "".join(train_letters)
        print(f"[INFO] Start 10-fold train (leave_out='{leave_out_letter}') for hyperparams: {hp_overrides or 'default'}")

        # データローダーから訓練データをイテレート
        model, optimizer = get_model(cfg, hp_overrides)
        Ny = cfg.model.Ny

        for sample_id, U, class_id in data_loader.load_fold_data(train_letters):
            T = len(U)
            D = make_onehot(class_id, T, Ny)
            self._trainer.train(model, optimizer, sample_id, U, D)

        weight_file_name = make_weight_filename(cfg=cfg, overrides=hp_overrides, train_tag=train_tag)
        dst = save_numpy_npy_atomic(
            model.Output.Wout,
            weight_dir,
            weight_file_name,
        )
        print(f"[INFO] Finished fold '{leave_out_letter}'. Weight saved to {dst}")

        end_time = time.monotonic()
        execution_time = end_time - start_time
        timestamp = datetime.now().isoformat()

        return execution_time, timestamp
