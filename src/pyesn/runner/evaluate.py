# runner/evaluate.py
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pyesn.setup.config import Config
from pyesn.pipeline.evaluator import Evaluator
from pyesn.pipeline.predictor import Predictor
from pyesn.model.model_builder import get_model
from pyesn.utils.data_processing import make_onehot
from pyesn.utils.io import load_jsonl, target_output_from_dict
from pyesn.runner.tenfold import utils as cv_utils


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
    - Aggregates metrics and appends a summary row per weight to evaluation_results.csv.
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

    # Prepare results CSV path
    results_csv = weight_dir / "evaluation_results.csv"

    # Iterate weights
    weight_files = sorted(weight_dir.glob("*_Wout.npy"))
    if not weight_files:
        print(f"[WARN] No weight files found in {weight_dir}")
        return

    rows = []
    for wf in weight_files:
        try:
            overrides, train_tag = _parse_weight_filename(wf)
        except Exception as e:
            print(f"[SKIP] {wf.name}: {e}")
            continue

        # Determine held-out fold
        train_set = set(train_tag)
        all_set = set(letters)
        holdouts = list(all_set - train_set)
        if len(holdouts) != 1:
            print(f"[SKIP] Could not determine a single held-out fold for {wf.name}")
            continue
        holdout = holdouts[0]

        # Build model and load weight
        model, _ = get_model(cfg, overrides)
        weight = np.load(wf, allow_pickle=True)
        model.Output.setweight(weight)
        print(f"[ARTIFACT] Loaded weight: {wf.name} | holdout='{holdout}' | hp={overrides}")

        # Load hold-out CSV data
        ids, paths, class_ids = cv_utils.read_data_from_csvs([csv_map[holdout]])
        assert len(ids) == len(paths) == len(class_ids), "length mismatch"

        # Run inference and aggregate metrics
        num_samples = len(paths)
        majority_success_count = 0
        total_correct = 0
        total_frames = 0

        for i in range(num_samples):
            img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {paths[i]}")
            U = img.T
            T = len(U)
            D = make_onehot(class_ids[i], T, cfg.model.Ny)

            record = predictor.predict(model, ids[i], U, D)

            # Majority-vote success per sample
            success, *_ = evaluator.majority_success(record)
            majority_success_count += int(success)

            # Per-timestep accuracy aggregation
            num_correct, acc, _ = evaluator.evaluate_classification_result(record)
            total_correct += num_correct
            total_frames += T

        majority_acc = (majority_success_count / num_samples) if num_samples > 0 else 0.0
        timestep_acc = (total_correct / total_frames) if total_frames > 0 else 0.0

        row = {
            "weight_file": wf.name,
            "train_folds": train_tag,
            "holdout_fold": holdout,
            "Nx": overrides["Nx"],
            "density": overrides["density"],
            "input_scale": overrides["input_scale"],
            "rho": overrides["rho"],
            "num_samples": num_samples,
            "majority_acc": round(float(majority_acc), 6),
            "timestep_acc": round(float(timestep_acc), 6),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        file_exists = results_csv.exists()
        df.to_csv(results_csv, mode='a', header=not file_exists, index=False)
        print(f"[INFO] Appended {len(rows)} evaluation rows to {results_csv}")
    else:
        print("[WARN] No evaluation results to save.")

    return
