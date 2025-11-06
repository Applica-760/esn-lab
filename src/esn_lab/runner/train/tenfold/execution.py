import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from .setup import init_global_worker_env, setup_worker_seed
from esn_lab.pipeline.train.tenfold_trainer import TenfoldTrainer
from esn_lab.utils.weight_management import WeightManager
from esn_lab.utils.execution_logging import ExecutionLogger


def execute_tasks(
    cfg,
    data_loader,
    weight_manager: WeightManager,
    execution_logger: ExecutionLogger,
    all_letters: list[str],
    hp_overrides: dict,
    tasks_to_run: list[str],
    parallel: bool,
    max_workers: int,
):
    """
    タスクリストに基づき、逐次または並列で学習を実行する。
    
    Args:
        cfg: 設定オブジェクト
        data_loader: データローダーインスタンス
        weight_manager: 重みファイル管理クラス
        execution_logger: 実行ログ記録クラス
        all_letters: 全fold ID一覧
        hp_overrides: ハイパーパラメータの上書き
        tasks_to_run: 実行するfold IDのリスト
        parallel: 並列実行するかどうか
        max_workers: 最大ワーカー数
    """
    if not parallel:
        _execute_sequentially(
            cfg, data_loader, weight_manager, execution_logger,
            all_letters, hp_overrides, tasks_to_run
        )
    else:
        _execute_in_parallel(
            cfg, data_loader, weight_manager, execution_logger,
            all_letters, hp_overrides, tasks_to_run, max_workers
        )

def _execute_sequentially(
    cfg, data_loader, weight_manager, execution_logger,
    all_letters, hp_overrides, tasks_to_run
):
    """タスクを逐次実行する。"""
    print(f"[INFO] Running {len(tasks_to_run)} tasks sequentially.")
    for i, leave in enumerate(tasks_to_run):
        try:
            _run_one_fold_search(
                cfg,
                data_loader=data_loader,
                weight_manager=weight_manager,
                execution_logger=execution_logger,
                all_letters=all_letters,
                leave_out_letter=leave,
                hp_overrides=hp_overrides,
                seed=i,
            )
        except Exception as e:
            print(f"[ERROR] Fold '{leave}' failed: {e}")

def _execute_in_parallel(
    cfg, data_loader, weight_manager, execution_logger,
    all_letters, hp_overrides, tasks_to_run, max_workers
):
    """タスクを並列実行する。"""
    print(f"[INFO] Running {len(tasks_to_run)} tasks in parallel with {max_workers} workers.")
    workers = min(max_workers, (os.cpu_count() or max_workers))
    executor_kwargs = {"max_workers": workers}
    if os.name == "posix":
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs, initializer=init_global_worker_env) as ex:
        future_to_leave = {
            ex.submit(
                _run_one_fold_search,
                cfg,
                data_loader=data_loader,
                weight_manager=weight_manager,
                execution_logger=execution_logger,
                all_letters=all_letters,
                leave_out_letter=leave,
                hp_overrides=hp_overrides,
                seed=i,
            ): leave
            for i, leave in enumerate(tasks_to_run)
        }

        for future in as_completed(future_to_leave):
            leave = future_to_leave[future]
            try:
                future.result()  # 例外が発生していないか確認
                print(f"[INFO] Fold '{leave}' completed successfully.")
            except Exception as e:
                print(f"[ERROR] Fold '{leave}' failed: {e}")


def _run_one_fold_search(
    cfg,
    data_loader,
    weight_manager: WeightManager,
    execution_logger: ExecutionLogger,
    all_letters: list[str],
    leave_out_letter: str,
    hp_overrides: dict,
    seed: int,
) -> None:
    """1-foldの学習を実行する（並列実行用のワーカー関数）。
    
    Runner層の責務: 乱数シード設定のみ。
    実際の学習処理はpipeline層（TenfoldTrainer）に委譲。
    
    Args:
        cfg: 設定オブジェクト
        data_loader: データローダーインスタンス
        weight_manager: 重みファイル管理クラス
        execution_logger: 実行ログ記録クラス
        all_letters: 全fold ID一覧
        leave_out_letter: テストfold ID
        hp_overrides: ハイパーパラメータ上書き
        seed: 乱数シード
    """
    # Runner層の責務: 乱数シード設定
    setup_worker_seed(seed)
    
    # Pipeline層に処理を委譲
    trainer = TenfoldTrainer(
        run_dir=cfg.run_dir,
        weight_manager=weight_manager,
        execution_logger=execution_logger,
    )
    trainer.run_one_fold_search(
        cfg=cfg,
        data_loader=data_loader,
        all_letters=all_letters,
        leave_out_letter=leave_out_letter,
        hp_overrides=hp_overrides,
    )