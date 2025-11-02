import os
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from esn_lab.setup.config import Config
from esn_lab.utils.io import load_jsonl, target_output_from_dict
from esn_lab.pipeline.eval.tenfold_evaluator import eval_one_weight_worker
from esn_lab.pipeline.tenfold_util import load_10fold_csv_mapping, parse_weight_filename, make_weight_filename
from esn_lab.pipeline.eval.evaluator import Evaluator
from esn_lab.runner.train.tenfold.setup import init_global_worker_env
from esn_lab.utils.param_grid import flatten_search_space


# per-weight 評価ロジックは pipeline 側に移行（eval_one_weight_worker）


def single_evaluate(cfg: Config):
    """Evaluate results already saved to predict_record.jsonl in a run directory."""
    run_dir = Path(cfg.evaluate.run.run_dir)

    file = list(run_dir.glob("predict_record.jsonl"))[0]
    datas = load_jsonl(file)

    evaluator = Evaluator()

    for i, data in enumerate(datas):
        record = target_output_from_dict(data)
        evaluator.majority_success(record)

    return


## Naming is centralized in pipeline.tenfold_util


def tenfold_evaluate(cfg: Config):
    """Evaluate tenfold-trained weights by inferring on the held-out fold for each weight.

    - Determines the held-out fold from the weight filename (train letters a-j).
    - Rebuilds the ESN with hyperparameters parsed from the filename.
    - Loads the corresponding CSV for the held-out fold and runs inference.
    - Appends a summary row per weight to evaluation_results.csv immediately after each weight.
    - If evaluation_results.csv already exists, skip weights that are already recorded.
    """
    # Ensure single-threaded math libs in child processes by setting env in parent before spawn
    # Note: initializer runs after worker process starts; some libs decide threads at import time.
    # Propagating these via the parent guarantees children import numpy/BLAS with 1 thread.
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = "1"

    ten_cfg = cfg.evaluate.tenfold
    if ten_cfg is None:
        raise ValueError("Config 'cfg.evaluate.tenfold' not found.")

    # Prepare paths
    csv_dir = Path(ten_cfg.csv_dir).expanduser().resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")

    weight_dir = (Path.cwd() / ten_cfg.weight_dir).resolve()
    if not weight_dir.exists():
        raise FileNotFoundError(f"weight_dir not found: {weight_dir}")

    # Prepare evaluation output directory alongside weight_dir
    out_dir = (weight_dir.parent / f"{weight_dir.name}_eval").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV mapping and letters
    csv_map = load_10fold_csv_mapping(csv_dir)
    letters = sorted(csv_map.keys())

    results_csv = out_dir / "evaluation_results.csv"
    processed_weights: set[str] = set()
    if results_csv.exists():
        try:
            prev_df = pd.read_csv(results_csv)
            if "weight_file" in prev_df.columns:
                processed_weights = set(prev_df["weight_file"].astype(str).tolist())
                print(f"[INFO] Found existing results for {len(processed_weights)} weights. Skipping duplicates.")
            else:
                print(f"[WARN] Existing CSV missing 'weight_file' column. Ignoring previous content: {results_csv}")
        except Exception as e:
            print(f"[WARN] Failed to read existing results CSV ({results_csv}): {e}. Proceeding without skip list.")

    tasks: list[tuple[Path, dict, str, str]] = []  # (wf_path, overrides, train_tag, holdout)

    # Require search_space to be specified; directory scanning fallback is removed
    if not getattr(ten_cfg, "search_space", None):
        raise ValueError("cfg.evaluate.tenfold.search_space is required (keys like 'model.Nx').")

    try:
        combos = flatten_search_space(ten_cfg.search_space)
    except Exception as e:
        raise ValueError(f"Invalid evaluate.tenfold.search_space: {e}")

    if not combos:
        print("[INFO] No parameter combos to evaluate.")
        return

    # for each combo and for each holdout, add existing weight path if available
    for overrides, _tag in combos:
        for holdout in letters:
            train_letters = [x for x in letters if x != holdout]
            train_tag = "".join(train_letters)
            fname = make_weight_filename(cfg=cfg, overrides=overrides, train_tag=train_tag)
            wf = weight_dir / fname
            if not wf.exists():
                # weight not trained yet; skip quietly
                continue
            if wf.name in processed_weights:
                # already evaluated
                continue
            tasks.append((wf, overrides, train_tag, holdout))

    if not tasks:
        print("[INFO] Nothing to evaluate (all weights already processed or skipped).")
        return

    # Prepare appender and decide parallelism
    ev_appender = Evaluator()
    workers = int(ten_cfg.workers or (os.cpu_count() or 1))
    do_parallel = bool(ten_cfg.parallel if ten_cfg.parallel is not None else True)
    if not do_parallel or workers <= 1:
        print(f"[INFO] Running {len(tasks)} evaluation tasks sequentially.")
        for (wf, overrides, train_tag, holdout) in tasks:
            try:
                row, pred_rows = eval_one_weight_worker(cfg, str(wf), str(csv_dir), overrides, train_tag, holdout)
                ev_appender.append_results(out_dir=out_dir, row=row, pred_rows=pred_rows)
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {wf.name}: {e}")
        return

    print(f"[INFO] Running {len(tasks)} evaluation tasks in parallel (workers={workers}).")
    executor_kwargs = {"max_workers": workers}
    if os.name == "posix":
        # forkだと中断/再開後にスレッド系ライブラリで高負荷やハングが起きやすいためspawnに切替
        executor_kwargs["mp_context"] = mp.get_context("spawn")

    with ProcessPoolExecutor(**executor_kwargs, initializer=init_global_worker_env) as ex:
        future_to_wf = {
            ex.submit(eval_one_weight_worker, cfg, str(wf), str(csv_dir), overrides, train_tag, holdout): wf
            for (wf, overrides, train_tag, holdout) in tasks
        }
        for fut in as_completed(future_to_wf):
            wf = future_to_wf[fut]
            try:
                row, pred_rows = fut.result()
                ev_appender.append_results(out_dir=out_dir, row=row, pred_rows=pred_rows)
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {wf.name}: {e}")

    return


def summary_evaluate(cfg: Config):
    # Delegate summary plotting (errorbar + confusion) to Evaluator（遅延importで親のcv2初期化を回避）
    from esn_lab.pipeline.eval.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.summarize(cfg)


# CSV追記は Evaluator.append_results に集約
