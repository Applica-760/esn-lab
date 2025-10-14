import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import task
from .setup import init_global_worker_env

def execute_tasks(cfg, env, hp_overrides, hp_tag, tasks_to_run, parallel, max_workers):
    """
    タスクリストに基づき、逐次または並列で学習を実行する。
    """
    if not parallel:
        _execute_sequentially(cfg, env, hp_overrides, tasks_to_run)
    else:
        _execute_in_parallel(cfg, env, hp_overrides, hp_tag, tasks_to_run, max_workers)

def _execute_sequentially(cfg, env, hp_overrides, tasks_to_run):
    """タスクを逐次実行する。"""
    print(f"[INFO] Running {len(tasks_to_run)} tasks sequentially.")
    for i, leave in enumerate(tasks_to_run):
        try:
            task.run_one_fold_search(
                cfg,
                csv_map=env["csv_map"],
                all_letters=env["letters"],
                leave_out_letter=leave,
                hp_overrides=hp_overrides,
                weight_dir=env["weight_dir"],
                seed=i # 逐次でもシードを渡して再現性を確保
            )
        except Exception as e:
            print(f"[ERROR] Fold '{leave}' failed: {e}")

def _execute_in_parallel(cfg, env, hp_overrides, hp_tag, tasks_to_run, max_workers):
    """タスクを並列実行する。"""
    print(f"[INFO] Running {len(tasks_to_run)} tasks in parallel.")
    workers = min(max_workers, (os.cpu_count() or max_workers))
    executor_kwargs = {"max_workers": workers}
    if os.name == "posix":
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs, initializer=init_global_worker_env) as ex:
        future_to_leave = {
            ex.submit(
                task.run_one_fold_search,
                cfg,
                csv_map=env["csv_map"],
                all_letters=env["letters"],
                leave_out_letter=leave,
                hp_overrides=hp_overrides,
                weight_dir=env["weight_dir"],
                seed=i
            ): leave for i, leave in enumerate(tasks_to_run)
        }

        for future in as_completed(future_to_leave):
            leave = future_to_leave[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Fold '{leave}' in combo '{hp_tag}' failed: {e}")