import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

from .setup import init_global_worker_env, setup_worker_seed
from esn_lab.pipeline.train.tenfold_trainer import TenfoldTrainer

def _append_result_to_csv(result: dict, weight_dir: Path):
    """
    1件の実行結果をDataFrameに変換し、CSVファイルに追記保存する。
    この関数はアトミックではないが、親プロセスからシーケンシャルに呼ばれるためロックは不要。
    """
    df = pd.DataFrame([result])
    df = df[["timestamp", "hp_tag", "fold", "execution_time_sec"]] # カラム順を整理
    
    # Save execution_times.csv at the common tenfold_integ root (parent of weights)
    csv_path = weight_dir.parent / "execution_times.csv"
    
    file_exists = csv_path.exists()
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False, float_format='%.4f')
    print(f"[INFO] Appended execution record to {csv_path.name} for fold '{result['fold']}'")

def execute_tasks(cfg, env, hp_overrides, hp_tag, tasks_to_run, parallel, max_workers):
    """
    タスクリストに基づき、逐次または並列で学習を実行する。
    fold完了ごとに実行結果をCSVに追記する。
    """
    if not parallel:
        _execute_sequentially(cfg, env, hp_overrides, hp_tag, tasks_to_run)
    else:
        _execute_in_parallel(cfg, env, hp_overrides, hp_tag, tasks_to_run, max_workers)

def _execute_sequentially(cfg, env, hp_overrides, hp_tag, tasks_to_run):
    """タスクを逐次実行し、完了ごとに結果を書き込む。"""
    print(f"[INFO] Running {len(tasks_to_run)} tasks sequentially.")
    for i, leave in enumerate(tasks_to_run):
        try:
            execution_time, timestamp = _run_one_fold_search(
                cfg,
                csv_map=env["csv_map"],
                all_letters=env["letters"],
                leave_out_letter=leave,
                hp_overrides=hp_overrides,
                weight_dir=env["weight_dir"],
                seed=i,
            )
            result = {
                "timestamp": timestamp,
                "hp_tag": hp_tag,
                "fold": leave,
                "execution_time_sec": execution_time,
            }
            _append_result_to_csv(result, env["weight_dir"])
        except Exception as e:
            print(f"[ERROR] Fold '{leave}' failed: {e}")

def _execute_in_parallel(cfg, env, hp_overrides, hp_tag, tasks_to_run, max_workers):
    """タスクを並列実行し、完了ごとに結果を書き込む。"""
    print(f"[INFO] Running {len(tasks_to_run)} tasks in parallel.")
    workers = min(max_workers, (os.cpu_count() or max_workers))
    executor_kwargs = {"max_workers": workers}
    if os.name == "posix":
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs, initializer=init_global_worker_env) as ex:
        future_to_leave = {
            ex.submit(
                _run_one_fold_search,
                cfg,
                csv_map=env["csv_map"],
                all_letters=env["letters"],
                leave_out_letter=leave,
                hp_overrides=hp_overrides,
                weight_dir=env["weight_dir"],
                seed=i,
            ): leave
            for i, leave in enumerate(tasks_to_run)
        }

        for future in as_completed(future_to_leave):
            leave = future_to_leave[future]
            try:
                execution_time, timestamp = future.result()
                result = {
                    "timestamp": timestamp,
                    "hp_tag": hp_tag,
                    "fold": leave,
                    "execution_time_sec": execution_time,
                }
                _append_result_to_csv(result, env["weight_dir"])
            except Exception as e:
                print(f"[ERROR] Fold '{leave}' in combo '{hp_tag}' failed: {e}")


def _run_one_fold_search(
    cfg,
    csv_map: dict[str, Path],
    all_letters: list[str],
    leave_out_letter: str,
    hp_overrides: dict,
    weight_dir: Path,
    seed: int,
) -> tuple[float, str]:
    """Thin wrapper: set seed (runner responsibility) and delegate to pipeline TenfoldTrainer."""
    setup_worker_seed(seed)
    trainer = TenfoldTrainer(cfg.run_dir)
    return trainer.run_one_fold_search(
        cfg=cfg,
        csv_map=csv_map,
        all_letters=all_letters,
        leave_out_letter=leave_out_letter,
        hp_overrides=hp_overrides,
        weight_dir=weight_dir,
    )