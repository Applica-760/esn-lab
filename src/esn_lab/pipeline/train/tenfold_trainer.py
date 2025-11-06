import time
from datetime import datetime
from pathlib import Path

from esn_lab.pipeline.train.trainer import Trainer
from esn_lab.pipeline.data import BaseDataLoader
from esn_lab.utils.weight_management import WeightManager
from esn_lab.utils.execution_logging import ExecutionLogger
from esn_lab.utils.data_processing import make_onehot
from esn_lab.model.model_builder import get_model, get_model_param_str
from esn_lab.utils.io import save_numpy_npy_atomic


class TenfoldTrainer:
    """10-fold 学習の処理本体(pipeline側)。

    - データ読み込み、モデル構築、学習、重み保存、実行ログ記録までを担当。
    - 乱数シード設定や並列実行の制御はrunner側の責務とする。
    """

    def __init__(
        self,
        run_dir: str,
        weight_manager: WeightManager,
        execution_logger: ExecutionLogger,
    ):
        """
        Args:
            run_dir: 実行ディレクトリ（Trainer用）
            weight_manager: 重みファイル管理クラス
            execution_logger: 実行ログ記録クラス
        """
        self._trainer = Trainer(run_dir)
        self._weight_manager = weight_manager
        self._execution_logger = execution_logger

    def run_one_fold_search(
        self,
        cfg,
        data_loader: BaseDataLoader,
        all_letters: list[str],
        leave_out_letter: str,
        hp_overrides: dict,
    ) -> None:
        """1-foldの学習を実行し、重みの保存と実行ログの記録を行う。
        
        Args:
            cfg: 設定オブジェクト
            data_loader: データローダーインスタンス
            all_letters: 全fold ID一覧
            leave_out_letter: テストfold ID（学習から除外）
            hp_overrides: ハイパーパラメータの上書き
        
        Raises:
            Exception: 学習中にエラーが発生した場合
        """
        start_time = time.monotonic()
        timestamp = datetime.now().isoformat()

        train_letters = [x for x in all_letters if x != leave_out_letter]
        train_tag = "".join(train_letters)
        hp_tag = get_model_param_str(cfg=cfg, overrides=hp_overrides)
        
        print(f"[INFO] Start 10-fold train (leave_out='{leave_out_letter}') for hyperparams: {hp_tag}")

        # データローダーから訓練データをイテレート
        model, optimizer = get_model(cfg, hp_overrides)
        Ny = cfg.model.Ny

        for sample_id, U, class_id in data_loader.load_fold_data(train_letters):
            T = len(U)
            D = make_onehot(class_id, T, Ny)
            self._trainer.train(model, optimizer, sample_id, U, D)

        # 重みファイルの保存先を取得
        weight_path = self._weight_manager.get_weight_path(cfg, hp_overrides, train_tag)
        
        # 重みを保存
        dst = save_numpy_npy_atomic(
            model.Output.Wout,
            weight_path.parent,
            weight_path.name,
        )
        print(f"[INFO] Finished fold '{leave_out_letter}'. Weight saved to {dst}")

        # 実行時間を計測
        end_time = time.monotonic()
        execution_time = end_time - start_time

        # 実行ログを記録
        self._execution_logger.log_execution(
            hp_tag=hp_tag,
            fold=leave_out_letter,
            execution_time_sec=execution_time,
            timestamp=timestamp,
        )
        print(f"[INFO] Execution time logged for fold '{leave_out_letter}': {execution_time:.2f}s")
