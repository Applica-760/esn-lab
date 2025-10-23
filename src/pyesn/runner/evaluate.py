# runner/evaluate.py
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from pyesn.setup.config import Config
from pyesn.pipeline.evaluator import Evaluator
from pyesn.pipeline.predictor import Predictor
from pyesn.model.model_builder import get_model
from pyesn.utils.io import load_jsonl, target_output_from_dict
from pyesn.runner.tenfold import utils as cv_utils
from pyesn.runner.tenfold.setup import init_global_worker_env


def _eval_one_weight(cfg: Config, weight_path: str, csv_dir: str, overrides: dict, train_tag: str, holdout: str) -> tuple[dict, list[dict]]:
    """Worker: evaluate one weight file on its holdout fold and return (row, pred_rows)."""
    from pyesn.pipeline.evaluator import Evaluator
    from pyesn.pipeline.predictor import Predictor
    from pyesn.model.model_builder import get_model
    from pyesn.runner.tenfold import utils as cv_utils
    import numpy as np
    import cv2

    # Build model and load weight
    model, _ = get_model(cfg, overrides)
    weight = np.load(weight_path, allow_pickle=True)
    model.Output.setweight(weight)

    # Load holdout data
    csv_map = cv_utils.load_10fold_csv_mapping(Path(csv_dir))
    ids, paths, class_ids = cv_utils.read_data_from_csvs([csv_map[holdout]])
    assert len(ids) == len(paths) == len(class_ids), "length mismatch"

    evaluator = Evaluator()
    predictor = Predictor(cfg.run_dir)

    row, pred_rows = evaluator.evaluate_dataset_majority(
        cfg=cfg,
        model=model,
        predictor=predictor,
        ids=ids,
        paths=paths,
        class_ids=class_ids,
        wf_name=Path(weight_path).name,
        train_tag=train_tag,
        holdout=holdout,
        overrides=overrides,
    )
    return row, pred_rows


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


def _decode_decimal(token: str) -> float:
    """Reconstruct a float value encoded by removing the decimal point.

    Example:
    - "09"   -> 0.9
    - "095"  -> 0.95
    - "0005" -> 0.0005
    - "0001" -> 0.001
    The rule is: int(token) / (10 ** (len(token) - 1)).
    """
    if not token or not token.isdigit():
        raise ValueError(f"Invalid decimal token: {token}")
    return int(token) / (10 ** (len(token) - 1))


def _parse_weight_filename(path: Path) -> tuple[dict, str]:
    """Parse weight filename and return (overrides, train_tag).

    Expected pattern (stem):
    seed-<seedid>_nx-<Nx>_density-<dd>_input_scale-<ii>_rho-<rr>_<trainletters>_Wout
    Where dd/ii/rr are numbers without decimal points, trainletters are 9 letters a-j.
    """
    stem = path.stem  # without .npy
    # Ensure trailing _Wout exists in stem
    if not stem.endswith("_Wout"):
        raise ValueError(f"Unexpected weight filename (no _Wout suffix): {path.name}")

    # Regex to capture fields
    # Example: seed-nonseed_nx-200_density-05_input_scale-0001_rho-09_abcdefghi_Wout
    pat = re.compile(
        r"seed-(?P<seed>[^_]+)"  # seed tag (ignored for overrides)
        r"_nx-(?P<nx>\d+)"
        r"_density-(?P<density>\d+)"
        r"_input_scale-(?P<input>\d+)"
        r"_rho-(?P<rho>\d+)"
        r"_(?P<train>[a-j]{9})"
        r"_Wout$"
    )
    m = pat.match(stem)
    if not m:
        raise ValueError(f"Unexpected weight filename format: {path.name}")

    nx = int(m.group("nx"))
    density = _decode_decimal(m.group("density"))
    input_scale = _decode_decimal(m.group("input"))
    rho = _decode_decimal(m.group("rho"))
    train_tag = m.group("train")

    overrides = {"Nx": nx, "density": density, "input_scale": input_scale, "rho": rho}
    return overrides, train_tag


def tenfold_evaluate(cfg: Config):
    """Evaluate tenfold-trained weights by inferring on the held-out fold for each weight.

    - Determines the held-out fold from the weight filename (train letters a-j).
    - Rebuilds the ESN with hyperparameters parsed from the filename.
    - Loads the corresponding CSV for the held-out fold and runs inference.
    - Appends a summary row per weight to evaluation_results.csv immediately after each weight.
    - If evaluation_results.csv already exists, skip weights that are already recorded.
    """
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

    # Load CSV mapping and letters
    csv_map = cv_utils.load_10fold_csv_mapping(csv_dir)
    letters = sorted(csv_map.keys())

    # Prepare evaluator/predictor
    evaluator = Evaluator()
    predictor = Predictor(cfg.run_dir)

    # Prepare results CSV path and already-processed set
    results_csv = weight_dir / "evaluation_results.csv"
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

    # Collect tasks (skip already processed)
    weight_files = sorted(weight_dir.glob("*_Wout.npy"))
    if not weight_files:
        print(f"[WARN] No weight files found in {weight_dir}")
        return

    tasks: list[tuple[Path, dict, str, str]] = []  # (wf_path, overrides, train_tag, holdout)
    for wf in weight_files:
        if wf.name in processed_weights:
            print(f"[SKIP] Already evaluated: {wf.name}")
            continue
        try:
            overrides, train_tag = _parse_weight_filename(wf)
        except Exception as e:
            print(f"[SKIP] {wf.name}: {e}")
            continue
        train_set = set(train_tag)
        all_set = set(letters)
        holdouts = list(all_set - train_set)
        if len(holdouts) != 1:
            print(f"[SKIP] Could not determine a single held-out fold for {wf.name}")
            continue
        holdout = holdouts[0]
        tasks.append((wf, overrides, train_tag, holdout))

    if not tasks:
        print("[INFO] Nothing to evaluate (all weights already processed or skipped).")
        return

    # Decide parallelism
    workers = int(ten_cfg.workers or (os.cpu_count() or 1))
    do_parallel = bool(ten_cfg.parallel if ten_cfg.parallel is not None else True)
    if not do_parallel or workers <= 1:
        print(f"[INFO] Running {len(tasks)} evaluation tasks sequentially.")
        for (wf, overrides, train_tag, holdout) in tasks:
            try:
                row, pred_rows = _eval_one_weight(cfg, str(wf), str(csv_dir), overrides, train_tag, holdout)
                evaluator.append_results(weight_dir=weight_dir, row=row, pred_rows=pred_rows)
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {wf.name}: {e}")
        return

    print(f"[INFO] Running {len(tasks)} evaluation tasks in parallel (workers={workers}).")
    executor_kwargs = {"max_workers": workers}
    if os.name == "posix":
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs, initializer=init_global_worker_env) as ex:
        future_to_wf = {
            ex.submit(_eval_one_weight, cfg, str(wf), str(csv_dir), overrides, train_tag, holdout): wf
            for (wf, overrides, train_tag, holdout) in tasks
        }
        for fut in as_completed(future_to_wf):
            wf = future_to_wf[fut]
            try:
                row, pred_rows = fut.result()
                evaluator.append_results(weight_dir=weight_dir, row=row, pred_rows=pred_rows)
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {wf.name}: {e}")

    return


def summary_evaluate(cfg: Config):
    # Delegate summary plotting (errorbar + confusion) to Evaluator
    evaluator = Evaluator()
    evaluator.summarize(cfg)
