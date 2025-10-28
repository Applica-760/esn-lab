from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from pyesn.pipeline.train.tenfold_trainer import TenfoldTrainer
from pyesn.pipeline.eval.tenfold_evaluator import TenfoldEvaluator
from pyesn.pipeline.tenfold_util.data import load_10fold_csv_mapping
from pyesn.pipeline.tenfold_util.naming import make_weight_filename


def _append_train_time(weight_dir: Path, hp_tag: str, fold: str, execution_time_sec: float, timestamp: str) -> None:
    """Append one training timing record to execution_times.csv in weight_dir."""
    csv_path = weight_dir / "execution_times.csv"
    df = pd.DataFrame([
        {
            "timestamp": timestamp,
            "hp_tag": hp_tag,
            "fold": fold,
            "execution_time_sec": execution_time_sec,
        }
    ])
    file_exists = csv_path.exists()
    df = df[["timestamp", "hp_tag", "fold", "execution_time_sec"]]
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False, float_format="%.4f")


def _load_processed_weights(results_csv: Path) -> set[str]:
    """Read evaluation_results.csv and return set of weight_file names already processed."""
    if not results_csv.exists():
        return set()
    try:
        prev = pd.read_csv(results_csv)
        if "weight_file" in prev.columns:
            return set(prev["weight_file"].astype(str).tolist())
    except Exception:
        pass
    return set()


def run_tenfold_integration(cfg) -> None:
    """Orchestrate 10-fold: train each fold then immediately evaluate the held-out fold.

    Expected config (cfg.integ.tenfold):
      - train.csv_dir, train.weight_path, train.search_space (optional)
      - eval.csv_dir, eval.weight_dir (通常は train.weight_path と同じ), eval.workers/parallel (未使用; 本実装は逐次)
    """

    # Reduce BLAS threads to avoid oversubscription in potential future parallelism
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = os.environ.get(_k, "1")

    # 1) Resolve config branches safely (dict or attr access tolerant)
    integ = getattr(cfg, "integ", None) if not isinstance(cfg, dict) else cfg.get("integ")
    if integ is None or (hasattr(integ, "tenfold") and getattr(integ, "tenfold") is None):
        raise ValueError("Config 'integ.tenfold' not found.")
    ten = integ["tenfold"] if isinstance(integ, dict) else getattr(integ, "tenfold")

    train_cfg = ten.get("train") if isinstance(ten, dict) else getattr(ten, "train")
    eval_cfg = ten.get("eval") if isinstance(ten, dict) else getattr(ten, "eval")
    if train_cfg is None or eval_cfg is None:
        raise ValueError("Config 'integ.tenfold.train' and 'integ.tenfold.eval' are required.")

    # 2) Prepare paths and mapping
    csv_dir = Path(train_cfg["csv_dir"] if isinstance(train_cfg, dict) else train_cfg.csv_dir).expanduser().resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")

    weight_dir_cfg = train_cfg.get("weight_path") if isinstance(train_cfg, dict) else getattr(train_cfg, "weight_path")
    weight_dir = (Path.cwd() / weight_dir_cfg).resolve()
    weight_dir.mkdir(parents=True, exist_ok=True)

    # evaluation output directory mirrors evaluate.tenfold behavior
    eval_weight_dir_cfg = eval_cfg.get("weight_dir") if isinstance(eval_cfg, dict) else getattr(eval_cfg, "weight_dir")
    if not eval_weight_dir_cfg:
        # default to training weight_dir if not given
        eval_weight_dir_cfg = str(weight_dir)
    out_dir = (Path(eval_weight_dir_cfg).resolve().parent / f"{Path(eval_weight_dir_cfg).resolve().name}_eval").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_map = load_10fold_csv_mapping(csv_dir)
    letters = sorted(csv_map.keys())

    # 3) Prepare search space
    search_space = train_cfg.get("search_space") if isinstance(train_cfg, dict) else getattr(train_cfg, "search_space")
    # flatten search space (reuse utils logic inline to avoid import cycle)
    combos: List[Tuple[Dict, str]] = []
    if not search_space:
        combos = [({}, "default")]
    else:
        from itertools import product
        items = []
        for k, vals in search_space.items():
            if not k.startswith("model."):
                raise ValueError(f"search_space key must start with 'model.': {k}")
            field = k.split(".", 1)[1]
            items.append((field, list(vals)))
        keys = [k for k, _ in items]
        lists = [v for _, v in items]
        for values in product(*lists):
            d = {k: v for k, v in zip(keys, values)}
            tag = "_".join([f"{k}={v}" for k, v in d.items()])
            combos.append((d, tag))

    # 4) Instantiate pipeline helpers
    run_dir = getattr(cfg, "run_dir", None) if not isinstance(cfg, dict) else cfg.get("run_dir")
    trainer = TenfoldTrainer(run_dir)
    evaluator = TenfoldEvaluator(run_dir)

    # 5) Evaluate skip list (already processed weights)
    results_csv = out_dir / "evaluation_results.csv"
    processed_weights = _load_processed_weights(results_csv)

    # 6) Iterate over hp combos and folds (sequential)
    for hp_overrides, hp_tag in combos:
        print("=" * 50)
        print(f"[INFO] Processing hyperparameter combo: {hp_tag}")
        print("=" * 50)

        for leave in letters:
            # Derive names/paths deterministically
            train_letters = [x for x in letters if x != leave]
            train_tag = "".join(train_letters)
            weight_file = make_weight_filename(cfg=cfg, overrides=hp_overrides, train_tag=train_tag)
            weight_path = weight_dir / weight_file

            # Train if weight missing
            if not weight_path.exists():
                print(f"[INFO] Train fold (holdout='{leave}') for combo '{hp_tag}' ...")
                t0 = time.monotonic()
                exec_time, ts = trainer.run_one_fold_search(
                    cfg=cfg,
                    csv_map=csv_map,
                    all_letters=letters,
                    leave_out_letter=leave,
                    hp_overrides=hp_overrides,
                    weight_dir=weight_dir,
                )
                _append_train_time(weight_dir, hp_tag, leave, exec_time, ts)
                print(f"[OK] Trained. Saved weight: {weight_path.name}")
            else:
                print(f"[SKIP] Weight exists. Skip training: {weight_path.name}")

            # Evaluate if not yet processed
            if weight_path.name in processed_weights:
                print(f"[SKIP] Already evaluated: {weight_path.name}")
                continue

            try:
                print(f"[INFO] Evaluate '{weight_path.name}' on holdout '{leave}' ...")
                row, pred_rows = evaluator.eval_weight_on_holdout(
                    cfg=cfg,
                    weight_path=weight_path,
                    csv_dir=csv_dir,
                    overrides=hp_overrides,
                    train_tag=train_tag,
                    holdout=leave,
                )

                # Append via Evaluator to keep a single CSV format
                from pyesn.pipeline.eval.evaluator import Evaluator
                Evaluator().append_results(out_dir=out_dir, row=row, pred_rows=pred_rows)
                processed_weights.add(weight_path.name)
                print(f"[OK] Evaluation appended: {out_dir / 'evaluation_results.csv'}")
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {weight_path.name}: {e}")

    print("=" * 50)
    print("[INFO] 10-fold train+evaluate integration finished.")
    print("=" * 50)
