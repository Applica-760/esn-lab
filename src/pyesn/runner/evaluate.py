# runner/evaluate.py
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
    # New: detailed per-sample prediction CSV for confusion matrices
    preds_csv = weight_dir / "evaluation_predictions.csv"
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

    # Iterate weights
    weight_files = sorted(weight_dir.glob("*_Wout.npy"))
    if not weight_files:
        print(f"[WARN] No weight files found in {weight_dir}")
        return

    for wf in weight_files:
        # Skip if already processed
        if wf.name in processed_weights:
            print(f"[SKIP] Already evaluated: {wf.name}")
            continue
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
        # New: collect per-sample majority predictions for confusion matrix
        pred_rows: list[dict] = []

        for i in range(num_samples):
            img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {paths[i]}")
            U = img.T
            T = len(U)
            D = make_onehot(class_ids[i], T, cfg.model.Ny)

            record = predictor.predict(model, ids[i], U, D)

            # Majority-vote success per sample (also get predicted/true majority labels)
            success, pred_major, true_major, _, _ = evaluator.majority_success(record)
            majority_success_count += int(success)

            # Collect a per-sample row for confusion matrix plotting later
            pred_rows.append({
                "weight_file": wf.name,
                "train_folds": train_tag,
                "holdout_fold": holdout,
                "Nx": overrides["Nx"],
                "density": overrides["density"],
                "input_scale": overrides["input_scale"],
                "rho": overrides["rho"],
                "sample_id": ids[i],
                "true_label": int(true_major),
                "pred_label": int(pred_major),
                "majority_success": bool(success),
            })

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
        # Append immediately to CSVs (create with header if not exists)
        try:
            df_row = pd.DataFrame([row])
            header_needed = not results_csv.exists()
            df_row.to_csv(results_csv, mode='a', header=header_needed, index=False)
            print(f"[INFO] Appended evaluation row to {results_csv}: {wf.name}")
        except Exception as e:
            print(f"[ERROR] Failed to append result for {wf.name} to CSV: {e}")

        # New: append per-sample prediction rows
        try:
            if pred_rows:
                df_preds = pd.DataFrame(pred_rows)
                header_needed_preds = not preds_csv.exists()
                df_preds.to_csv(preds_csv, mode='a', header=header_needed_preds, index=False)
                print(f"[INFO] Appended {len(pred_rows)} prediction rows to {preds_csv}: {wf.name}")
        except Exception as e:
            print(f"[ERROR] Failed to append prediction rows for {wf.name} to CSV: {e}")

    return


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for key, val in filters.items():
        if key not in out.columns:
            print(f"[WARN] Filter column not found: {key}. Skipped.")
            continue
        series = out[key]
        if pd.api.types.is_numeric_dtype(series) and isinstance(val, (int, float)):
            # numeric tolerance filter
            tol = 1e-9
            out = out[(series - float(val)).abs() < tol]
        else:
            out = out[series.astype(str) == str(val)]
    return out


def summary_evaluate(cfg: Config):
    """Summarize evaluation_results.csv and also plot confusion matrices per vary value.

        Error-bar summary:
            - Uses evaluation_results.csv (one row per weight) to plot mean±std of a chosen metric.

        Confusion matrices:
            - If evaluation_predictions.csv exists, builds a confusion matrix aggregated across all matching rows
                for each vary value under the same filters. Saves one confusion matrix per vary value.

        Config fields (cfg.evaluate.summary):
            - weight_dir: directory containing evaluation_results.csv (+ evaluation_predictions.csv)
            - csv_name: file name (default: evaluation_results.csv)
            - metric: "majority_acc" or "timestep_acc"
            - vary_param: column to vary on x-axis (e.g., "Nx")
            - vary_values: list of values for vary_param (e.g., [200,300,400]); if None, use unique values in CSV
            - filters: dict of column->value to subset before aggregation (do NOT include vary_param)
            - agg: aggregation function name (kept for compatibility; not used here)
            - output_dir: directory to save plots (default: weight_dir/evaluation_plots)
            - fmt, title, dpi: output options
        """
    sum_cfg = cfg.evaluate.summary
    if sum_cfg is None:
        raise ValueError("Config 'cfg.evaluate.summary' not found.")

    weight_dir = Path(sum_cfg.weight_dir or ".").expanduser().resolve()
    csv_path = (weight_dir / (sum_cfg.csv_name or "evaluation_results.csv")).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    metric = sum_cfg.metric or "majority_acc"
    vary_param = sum_cfg.vary_param or "Nx"

    required_cols = {metric, vary_param}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

    # Apply filters
    filters = dict(sum_cfg.filters) if sum_cfg.filters else {}
    if vary_param in filters:
        print(f"[WARN] filters contains vary_param '{vary_param}'. Removing it from filters for aggregation.")
        filters.pop(vary_param, None)
    df_f = _apply_filters(df, filters)
    if df_f.empty:
        raise ValueError("No rows remain after applying filters. Adjust cfg.evaluate.summary.filters")

    # Determine values to iterate on
    if sum_cfg.vary_values is not None:
        vary_values = list(sum_cfg.vary_values)
    else:
        # Use unique values from data
        vary_values = sorted(df_f[vary_param].dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))

    # Aggregate mean and std over folds per vary value
    xs, means, stds, counts = [], [], [], []
    for v in vary_values:
        series = df_f[vary_param]
        if pd.api.types.is_numeric_dtype(series) and isinstance(v, (int, float)):
            tol = 1e-9
            df_v = df_f[(series - float(v)).abs() < tol]
        else:
            df_v = df_f[series.astype(str) == str(v)]

        vals = df_v[metric].dropna().astype(float)
        if len(vals) == 0:
            print(f"[WARN] No rows for {vary_param}={v} after filtering. Skipping this point.")
            continue
        xs.append(v)
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=0)))  # population std for 10 folds
        counts.append(int(vals.shape[0]))

    if not xs:
        raise ValueError("No data points to plot after filtering and varying parameter selection.")

    # Prepare output directory
    out_dir = Path(sum_cfg.output_dir).expanduser().resolve() if sum_cfg.output_dir else (weight_dir / "evaluation_plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot error bars (no connecting line; lighter error bars)
    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(xs)), 4))
    marker_style = dict(fmt='o', linestyle='none', markersize=6, markerfacecolor='C0', markeredgecolor='C0')
    # Use default error bar color (revert color to original behavior), keep thinner line
    ax.errorbar(xs, means, yerr=stds, capsize=4, elinewidth=1.0, **marker_style)
    ax.set_xlabel(vary_param)
    ax.set_ylabel(metric)
    title = sum_cfg.title or f"{metric} vs {vary_param} (mean±std)"
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    # Set y-axis limits if desired; using set_ylim (not get_ylim)
    ax.set_ylim(0.78, 0.96)
    fig.tight_layout()

    # Save outputs
    fname_base = f"errorbar_{metric}_by_{vary_param}"
    png_path = out_dir / f"{fname_base}.png"
    fig.savefig(png_path, dpi=int(sum_cfg.dpi or 150))
    plt.close(fig)
    print(f"[ARTIFACT] Saved errorbar plot: {png_path}")

    # Also save the aggregated table
    out_df = pd.DataFrame({vary_param: xs, f"{metric}_mean": means, f"{metric}_std": stds, "count": counts})
    csv_path = out_dir / f"{fname_base}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"[ARTIFACT] Saved summary CSV: {csv_path}")

    # ==============================
    # Confusion matrices (optional)
    # ==============================
    preds_csv = (weight_dir / "evaluation_predictions.csv").resolve()
    if not preds_csv.exists():
        print(f"[WARN] Predictions CSV not found; skipping confusion matrices: {preds_csv}")
        return

    try:
        dfp = pd.read_csv(preds_csv)
    except Exception as e:
        print(f"[WARN] Failed to read predictions CSV ({preds_csv}): {e}. Skipping confusion matrices.")
        return

    # Sanity check of required columns
    required_pred_cols = {"true_label", "pred_label", vary_param}
    missing_pred = [c for c in required_pred_cols if c not in dfp.columns]
    if missing_pred:
        print(f"[WARN] Missing columns in predictions CSV: {missing_pred}. Skipping confusion matrices.")
        return

    # Apply the same filters to predictions
    dfp_f = _apply_filters(dfp, filters)
    if dfp_f.empty:
        print("[WARN] No prediction rows remain after applying filters. Skipping confusion matrices.")
        return

    # Determine class count; prefer config if available
    try:
        n_classes = int(cfg.num_of_classes)
    except Exception:
        # Fallback to inferred max label + 1
        n_classes = int(max(dfp_f[["true_label", "pred_label"]].max()) + 1)

    # Generate one confusion matrix per vary value (using the same xs order)
    for v in xs:
        series = dfp_f[vary_param]
        if pd.api.types.is_numeric_dtype(series) and isinstance(v, (int, float)):
            tol = 1e-9
            df_v = dfp_f[(series - float(v)).abs() < tol]
        else:
            df_v = dfp_f[series.astype(str) == str(v)]

        if df_v.empty:
            print(f"[WARN] No prediction rows for {vary_param}={v}. Skipping confusion plot.")
            continue

        # Build confusion matrix: rows=true, cols=pred
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for _, rowp in df_v.iterrows():
            t = int(rowp["true_label"]) ; p = int(rowp["pred_label"]) 
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t, p] += 1

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f"Confusion matrix: {vary_param}={v}")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))

        # Annotate counts
        thresh = cm.max() / 2 if cm.max() > 0 else 0.5
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                ax.text(j, i, str(val), ha='center', va='center', color='white' if val > thresh else 'black')

        fig.tight_layout()
        fname_base_cm = f"confusion_{vary_param}-{v}"
        png_cm = out_dir / f"{fname_base_cm}.png"
        fig.savefig(png_cm, dpi=int(sum_cfg.dpi or 150))
        plt.close(fig)
        print(f"[ARTIFACT] Saved confusion matrix: {png_cm}")

        # Save raw confusion counts as CSV
        csv_cm = out_dir / f"{fname_base_cm}.csv"
        pd.DataFrame(cm).to_csv(csv_cm, index=False, header=[f"pred_{i}" for i in range(n_classes)])
        print(f"[ARTIFACT] Saved confusion counts CSV: {csv_cm}")

    return
