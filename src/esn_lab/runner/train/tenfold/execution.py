import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from .setup import init_global_worker_env, setup_worker_seed
from esn_lab.pipeline.train.tenfold_trainer import TenfoldTrainer
from esn_lab.utils.weight_management import WeightManager
from esn_lab.utils.execution_logging import ExecutionLogger


def execute_tasks(
    cfg,
    tenfold_cfg,
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
        tenfold_cfg: TrainTenfoldCfg オブジェクト
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
            cfg, tenfold_cfg, data_loader, weight_manager, execution_logger,
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
    cfg, tenfold_cfg, data_loader, weight_manager, execution_logger,
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
                _run_one_fold_search_parallel,
                cfg,
                tenfold_cfg=tenfold_cfg,
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
    
    # 並列実行完了後、個別のログファイルを統合
    _merge_execution_logs(execution_logger)


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
    """1-foldの学習を実行する（逐次実行用）。
    
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


def _run_one_fold_search_parallel(
    cfg,
    tenfold_cfg,
    all_letters: list[str],
    leave_out_letter: str,
    hp_overrides: dict,
    seed: int,
) -> None:
    """1-foldの学習を実行する（並列実行用）。
    
    並列処理では pickle できないオブジェクト（data_loader等）を渡せないため、
    各ワーカープロセス内で必要なオブジェクトを再作成する。
    
    各ワーカーはプロセスIDを含むログファイル名で実行時間を記録し、
    後でメインプロセスが統合する。
    
    Args:
        cfg: 設定オブジェクト
        tenfold_cfg: TrainTenfoldCfg オブジェクト
        all_letters: 全fold ID一覧
        leave_out_letter: テストfold ID
        hp_overrides: ハイパーパラメータ上書き
        seed: 乱数シード
    """
    from pathlib import Path
    from esn_lab.pipeline.data import create_data_loader_from_config
    from esn_lab.utils.weight_management import WeightManager
    from esn_lab.utils.execution_logging import ExecutionLogger
    
    # Runner層の責務: 乱数シード設定
    setup_worker_seed(seed)
    
    # 各ワーカープロセスで必要なオブジェクトを再作成
    if tenfold_cfg is None:
        raise ValueError("tenfold_cfg is required for parallel execution.")
    
    experiment_name = getattr(tenfold_cfg, "experiment_name", None)
    if not experiment_name:
        raise ValueError("Config requires 'train.tenfold.experiment_name'.")
    
    experiment_dir = Path("outputs/experiments") / experiment_name
    experiment_dir = experiment_dir.expanduser().resolve()
    
    weight_dir = experiment_dir / "weights"
    
    # データローダーと管理オブジェクトを再作成
    data_loader, _ = create_data_loader_from_config(cfg, tenfold_cfg)
    weight_manager = WeightManager(weight_dir)
    
    # 並列実行時はプロセスIDを含むログファイル名で記録
    pid = os.getpid()
    execution_logger = ExecutionLogger(
        experiment_dir,
        log_filename=f"execution_times_{pid}.csv"
    )
    
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


def _merge_execution_logs(execution_logger: ExecutionLogger) -> None:
    """並列実行で生成された個別のログファイルを統合する。
    
    各ワーカープロセスが作成した execution_times_{pid}.csv を
    メインの execution_times.csv に統合し、個別ファイルは削除する。
    
    Args:
        execution_logger: メインプロセスのExecutionLoggerインスタンス
    """
    import pandas as pd
    from pathlib import Path
    
    log_dir = execution_logger.log_path.parent
    main_log_path = execution_logger.log_path
    
    # execution_times_{pid}.csv パターンのファイルを検索
    worker_logs = list(log_dir.glob("execution_times_*.csv"))
    
    if not worker_logs:
        print("[INFO] No worker log files found to merge.")
        return
    
    print(f"[INFO] Merging {len(worker_logs)} worker log files into {main_log_path.name}...")
    
    # すべてのワーカーログを読み込んで結合
    dfs = []
    for worker_log in worker_logs:
        try:
            df = pd.read_csv(worker_log)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {worker_log.name}: {e}")
    
    if not dfs:
        print("[WARN] No valid worker logs to merge.")
        return
    
    # DataFrameを結合
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # タイムスタンプでソート
    if "timestamp" in merged_df.columns:
        merged_df = merged_df.sort_values("timestamp")
    
    # メインログファイルに書き込み
    try:
        # 既存のメインログがあれば読み込んで結合
        if main_log_path.exists():
            existing_df = pd.read_csv(main_log_path)
            merged_df = pd.concat([existing_df, merged_df], ignore_index=True)
            if "timestamp" in merged_df.columns:
                merged_df = merged_df.sort_values("timestamp")
        
        merged_df.to_csv(main_log_path, index=False, float_format='%.4f')
        print(f"[INFO] Successfully merged {len(dfs)} log files into {main_log_path.name}")
        
        # ワーカーログファイルを削除
        for worker_log in worker_logs:
            try:
                worker_log.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete {worker_log.name}: {e}")
        
    except Exception as e:
        print(f"[ERROR] Failed to merge logs: {e}")
        print("[INFO] Worker log files will be kept for manual inspection.")