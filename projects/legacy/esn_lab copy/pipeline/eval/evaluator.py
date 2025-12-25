# pipeline/evaluator.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from esn_lab.setup.config import TargetOutput, Config
from esn_lab.model.esn import ESN
from esn_lab.pipeline.pred.predictor import Predictor
from esn_lab.utils.data_processing import make_onehot
from esn_lab.utils.eval_utils import apply_filters
from esn_lab.utils.plotting import plot_errorbar_and_save, plot_confusion_matrices_and_save



class Evaluator:

    # 毎時刻評価
    def evaluate_classification_result(self, record: TargetOutput):
        Y = np.array(record.data.output_series)
        D = np.array(record.data.target_series)
        pred_idx = Y.argmax(axis=1) # モデル出力の最大値インデックス番号を取得
        T = D.shape[0]  # 時系列長を取得

        # モデル予測と正解ラベルが一致していたらTrue
        correct_mask = D[np.arange(T), pred_idx].astype(bool)   

        num_correct = int(correct_mask.sum())
        acc = float(correct_mask.mean())
        return num_correct, acc, correct_mask
    

    # 従来多数決評価
    def majority_success(self, record: TargetOutput):
        Y = np.array(record.data.output_series)
        D = np.array(record.data.target_series)
        pred_idx = Y.argmax(axis=1)
        true_idx = D.argmax(axis=1)

        C = D.shape[1]

        # Validate indices are within valid range
        if len(pred_idx) > 0 and (pred_idx.min() < 0 or pred_idx.max() >= C):
            raise ValueError(
                f"pred_idx contains out-of-range values: min={pred_idx.min()}, max={pred_idx.max()}, "
                f"expected range [0, {C-1}]"
            )
        if len(true_idx) > 0 and (true_idx.min() < 0 or true_idx.max() >= C):
            raise ValueError(
                f"true_idx contains out-of-range values: min={true_idx.min()}, max={true_idx.max()}, "
                f"expected range [0, {C-1}]"
            )

        pred_counts = np.bincount(pred_idx, minlength=C)
        true_counts = np.bincount(true_idx, minlength=C)

        pred_major = int(pred_counts.argmax())
        true_major = int(true_counts.argmax())

        success = (pred_major == true_major)
        return success, pred_major, true_major, pred_counts, true_counts


    def make_confusion_matrix(self):
        return

    # ==============================
    # Dataset evaluation helpers
    # ==============================
    def evaluate_dataset_majority(
        self,
        cfg: Config,
        model: ESN,
        predictor: Predictor,
        ids: list[str],
        sequences: list[np.ndarray],
        class_ids: list[int],
        wf_name: str,
        train_tag: str,
        holdout: str,
        overrides: dict,
    ) -> tuple[dict, list[dict]]:
        """Evaluate a dataset (list of sequences) and return summary row and per-sample majority rows.

        Args:
            sequences: リスト of (T, D) numpy arrays - 時系列データ
        
        Returns:
          - row: dict for one line in evaluation_results.csv
          - pred_rows: list of dicts for evaluation_predictions.csv (per-sample true/pred majority)
        """
        num_samples = len(sequences)
        majority_success_count = 0
        total_correct = 0
        total_frames = 0
        pred_rows: list[dict] = []
        
        # Class-wise counters for robust evaluation (dynamic for any number of classes)
        num_classes = int(cfg.model.Ny)
        class_wise_correct = {cls: 0 for cls in range(num_classes)}
        class_wise_total = {cls: 0 for cls in range(num_classes)}

        for i in range(num_samples):
            U = sequences[i]
            T = len(U)
            expected_label = int(class_ids[i])
            D = make_onehot(expected_label, T, cfg.model.Ny)

            record = predictor.predict(model, ids[i], U, D)

            # Majority per-sample
            success, pred_major, true_major, _, _ = self.majority_success(record)
            
            # Validate that true_major derived from D matches expected_label
            if true_major != expected_label:
                raise ValueError(
                    f"Label mismatch for sample {ids[i]}: "
                    f"class_ids[{i}]={expected_label} but true_major={true_major}. "
                    f"This indicates data corruption or incorrect one-hot encoding."
                )
            
            # Track class-wise accuracy
            if expected_label not in class_wise_total:
                raise ValueError(
                    f"Unexpected label {expected_label} for sample {ids[i]}: "
                    f"valid range is 0 to {num_classes-1}. "
                    f"This indicates data corruption or incorrect class_id."
                )
            class_wise_total[expected_label] += 1
            if success:
                class_wise_correct[expected_label] += 1
            
            majority_success_count += int(success)

            pred_rows.append({
                "weight_file": wf_name,
                "train_folds": train_tag,
                "holdout_fold": holdout,
                "Nx": overrides.get("Nx"),
                "density": overrides.get("density"),
                "input_scale": overrides.get("input_scale"),
                "rho": overrides.get("rho"),
                "sample_id": ids[i],
                "expected_label": expected_label,  # From class_ids
                "true_label": int(true_major),     # From D (should match expected_label)
                "pred_label": int(pred_major),     # Model prediction
                "majority_success": bool(success),
            })

            # Per-timestep
            num_correct, acc, _ = self.evaluate_classification_result(record)
            total_correct += num_correct
            total_frames += T

        majority_acc = (majority_success_count / num_samples) if num_samples > 0 else 0.0
        timestep_acc = (total_correct / total_frames) if total_frames > 0 else 0.0
        
        # Compute class-wise accuracies (dynamic for all classes)
        class_accs = {}
        for cls in range(num_classes):
            total = class_wise_total[cls]
            correct = class_wise_correct[cls]
            class_accs[f"class_{cls}_acc"] = round(float(correct / total), 6) if total > 0 else None
            class_accs[f"class_{cls}_count"] = total

        row = {
            "weight_file": wf_name,
            "train_folds": train_tag,
            "holdout_fold": holdout,
            "Nx": overrides.get("Nx"),
            "density": overrides.get("density"),
            "input_scale": overrides.get("input_scale"),
            "rho": overrides.get("rho"),
            "num_samples": num_samples,
            "majority_acc": round(float(majority_acc), 6),
            "timestep_acc": round(float(timestep_acc), 6),
            **class_accs,  # Add class-wise metrics
        }

        return row, pred_rows

    def append_results(
        self,
        out_dir: Path,
        row: dict,
        pred_rows: list[dict],
    ):
        """Append a summary row and per-sample prediction rows to CSVs in the given output directory.

        Note: The caller should provide the evaluation output directory (sibling of weight_dir),
        not the weight directory itself.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        results_csv = out_dir / "evaluation_results.csv"
        preds_csv = out_dir / "evaluation_predictions.csv"

        try:
            df_row = pd.DataFrame([row])
            header_needed = not results_csv.exists()
            df_row.to_csv(results_csv, mode='a', header=header_needed, index=False)
            print(f"[INFO] Appended evaluation row to {results_csv}: {row.get('weight_file')}")
        except Exception as e:
            print(f"[ERROR] Failed to append result for {row.get('weight_file')} to CSV: {e}")

        try:
            if pred_rows:
                df_preds = pd.DataFrame(pred_rows)
                header_needed_preds = not preds_csv.exists()
                df_preds.to_csv(preds_csv, mode='a', header=header_needed_preds, index=False)
                print(f"[INFO] Appended {len(pred_rows)} prediction rows to {preds_csv}: {row.get('weight_file')}")
        except Exception as e:
            print(f"[ERROR] Failed to append prediction rows for {row.get('weight_file')} to CSV: {e}")


    # ==============================
    # Summary/plot helpers
    # ==============================
    def summarize(self, cfg: Config):
        sum_cfg = cfg.evaluate.summary
        if sum_cfg is None:
            raise ValueError("Config 'cfg.evaluate.summary' not found.")

        # パス解決の優先順位:
        # 1. results_csv が明示的に指定されている
        # 2. experiment_name から自動補完（推奨）
        
        results_csv_explicit = getattr(sum_cfg, "results_csv", None)
        predictions_csv_explicit = getattr(sum_cfg, "predictions_csv", None)
        experiment_name = getattr(sum_cfg, "experiment_name", None)
        
        # results_csv の決定
        if results_csv_explicit:
            csv_path = Path(results_csv_explicit).expanduser().resolve()
        elif experiment_name:
            exp_base = Path("outputs/experiments") / experiment_name / "eval"
            csv_path = (exp_base / "evaluation_results.csv").resolve()
            print(f"[INFO] Using experiment: {experiment_name}")
        else:
            raise ValueError("Config requires either 'results_csv' or 'experiment_name'.")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"results CSV not found: {csv_path}")
        
        # predictions_csv の決定（confusion matrix用）
        if predictions_csv_explicit:
            preds_csv = Path(predictions_csv_explicit).expanduser().resolve()
        elif experiment_name:
            exp_base = Path("outputs/experiments") / experiment_name / "eval"
            preds_csv = (exp_base / "evaluation_predictions.csv").resolve()
        else:
            preds_csv = csv_path.parent / "evaluation_predictions.csv"
        
        # 出力ディレクトリの決定
        if hasattr(sum_cfg, "output_dir") and sum_cfg.output_dir:
            images_dir = Path(sum_cfg.output_dir).expanduser().resolve()
        elif experiment_name:
            images_dir = (Path("outputs/experiments") / experiment_name / "eval" / "images").resolve()
        else:
            images_dir = csv_path.parent / "images"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = csv_path.parent / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

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
        df_f = apply_filters(df, filters)
        if df_f.empty:
            raise ValueError("No rows remain after applying filters. Adjust cfg.evaluate.summary.filters")

        # Determine values to iterate on
        if sum_cfg.vary_values is not None:
            vary_values = list(sum_cfg.vary_values)
        else:
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
            stds.append(float(vals.std(ddof=0)))
            counts.append(int(vals.shape[0]))

        if not xs:
            raise ValueError("No data points to plot after filtering and varying parameter selection.")

        # Plot error bars via utility
        fname_base = f"errorbar_{metric}_by_{vary_param}"
        _png, _csv = plot_errorbar_and_save(
            xs=xs,
            means=means,
            stds=stds,
            vary_param=vary_param,
            metric=metric,
            title=sum_cfg.title,
            out_dir=images_dir,
            dpi=int(sum_cfg.dpi or 150),
            ylim=(0.70, 0.90),
            counts=counts,
            fname_base=fname_base,
        )
        # Move the aggregated CSV to csv_dir
        try:
            target_csv = csv_dir / _csv.name
            _csv.replace(target_csv)
            print(f"[ARTIFACT] Moved summary CSV to: {target_csv}")
        except Exception as e:
            print(f"[WARN] Failed to move summary CSV to csv_dir: {e}")

        # Confusion matrices (optional)
        if not preds_csv.exists():
            print(f"[WARN] Predictions CSV not found; skipping confusion matrices: {preds_csv}")
            return

        try:
            dfp = pd.read_csv(preds_csv)
        except Exception as e:
            print(f"[WARN] Failed to read predictions CSV ({preds_csv}): {e}. Skipping confusion matrices.")
            return

        required_pred_cols = {"true_label", "pred_label", vary_param}
        missing_pred = [c for c in required_pred_cols if c not in dfp.columns]
        if missing_pred:
            print(f"[WARN] Missing columns in predictions CSV: {missing_pred}. Skipping confusion matrices.")
            return

        dfp_f = apply_filters(dfp, filters)
        if dfp_f.empty:
            print("[WARN] No prediction rows remain after applying filters. Skipping confusion matrices.")
            return

        try:
            n_classes = int(cfg.num_of_classes)
        except Exception:
            n_classes = int(max(dfp_f[["true_label", "pred_label"]].max()) + 1)

        # Confusion matrices are row-normalized only (each true-label row sums to 1)
        plot_confusion_matrices_and_save(
            dfp=dfp_f,
            xs=xs,
            vary_param=vary_param,
            n_classes=n_classes,
            out_dir=images_dir,
            dpi=int(sum_cfg.dpi or 150),
            base_prefix="confusion_",
        )
        # Move confusion CSVs to csv_dir
        try:
            for p in images_dir.glob("confusion_*.csv"):
                target = csv_dir / p.name
                p.replace(target)
                print(f"[ARTIFACT] Moved confusion CSV to: {target}")
        except Exception as e:
            print(f"[WARN] Failed to move confusion CSVs: {e}")

        return

