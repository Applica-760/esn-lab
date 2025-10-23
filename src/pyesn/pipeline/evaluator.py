# pipeline/evaluator.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from pyesn.setup.config import TargetOutput, Config
from pyesn.model.esn import ESN
from pyesn.model.model_builder import get_model, get_model_param_str
from pyesn.pipeline.predictor import Predictor
from pyesn.utils.data_processing import make_onehot



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
        paths: list[str],
        class_ids: list[int],
        wf_name: str,
        train_tag: str,
        holdout: str,
        overrides: dict,
    ) -> tuple[dict, list[dict]]:
        """Evaluate a dataset (list of image paths) and return summary row and per-sample majority rows.

        Returns:
          - row: dict for one line in evaluation_results.csv
          - pred_rows: list of dicts for evaluation_predictions.csv (per-sample true/pred majority)
        """
        num_samples = len(paths)
        majority_success_count = 0
        total_correct = 0
        total_frames = 0
        pred_rows: list[dict] = []

        for i in range(num_samples):
            img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {paths[i]}")
            U = img.T
            T = len(U)
            D = make_onehot(class_ids[i], T, cfg.model.Ny)

            record = predictor.predict(model, ids[i], U, D)

            # Majority per-sample
            success, pred_major, true_major, _, _ = self.majority_success(record)
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
                "true_label": int(true_major),
                "pred_label": int(pred_major),
                "majority_success": bool(success),
            })

            # Per-timestep
            num_correct, acc, _ = self.evaluate_classification_result(record)
            total_correct += num_correct
            total_frames += T

        majority_acc = (majority_success_count / num_samples) if num_samples > 0 else 0.0
        timestep_acc = (total_correct / total_frames) if total_frames > 0 else 0.0

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
        }

        return row, pred_rows

    def append_results(
        self,
        weight_dir: Path,
        row: dict,
        pred_rows: list[dict],
    ):
        """Append a summary row and per-sample prediction rows to CSVs in weight_dir."""
        results_csv = weight_dir / "evaluation_results.csv"
        preds_csv = weight_dir / "evaluation_predictions.csv"

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
    @staticmethod
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
                tol = 1e-9
                out = out[(series - float(val)).abs() < tol]
            else:
                out = out[series.astype(str) == str(val)]
        return out

    def summarize(self, cfg: Config):
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
        df_f = self._apply_filters(df, filters)
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

        # Prepare output directory
        out_dir = Path(sum_cfg.output_dir).expanduser().resolve() if sum_cfg.output_dir else (weight_dir / "evaluation_plots").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Plot error bars
        fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(xs)), 4))
        marker_style = dict(fmt='o', linestyle='none', markersize=6, markerfacecolor='C0', markeredgecolor='C0')
        ax.errorbar(xs, means, yerr=stds, capsize=4, elinewidth=1.0, **marker_style)
        ax.set_xlabel(vary_param)
        ax.set_ylabel(metric)
        title = sum_cfg.title or f"{metric} vs {vary_param} (mean±std)"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(0.78, 0.96)
        fig.tight_layout()

        fname_base = f"errorbar_{metric}_by_{vary_param}"
        png_path = out_dir / f"{fname_base}.png"
        fig.savefig(png_path, dpi=int(sum_cfg.dpi or 150))
        plt.close(fig)
        print(f"[ARTIFACT] Saved errorbar plot: {png_path}")

        # Also save aggregated table
        out_df = pd.DataFrame({vary_param: xs, f"{metric}_mean": means, f"{metric}_std": stds, "count": counts})
        csv_path2 = out_dir / f"{fname_base}.csv"
        out_df.to_csv(csv_path2, index=False)
        print(f"[ARTIFACT] Saved summary CSV: {csv_path2}")

        # Confusion matrices (optional)
        preds_csv = (weight_dir / "evaluation_predictions.csv").resolve()
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

        dfp_f = self._apply_filters(dfp, filters)
        if dfp_f.empty:
            print("[WARN] No prediction rows remain after applying filters. Skipping confusion matrices.")
            return

        try:
            n_classes = int(cfg.num_of_classes)
        except Exception:
            n_classes = int(max(dfp_f[["true_label", "pred_label"]].max()) + 1)

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

            cm = np.zeros((n_classes, n_classes), dtype=int)
            for _, rowp in df_v.iterrows():
                t = int(rowp["true_label"]) ; p = int(rowp["pred_label"])
                if 0 <= t < n_classes and 0 <= p < n_classes:
                    cm[t, p] += 1

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title(f"Confusion matrix: {vary_param}={v}")
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))

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

            csv_cm = out_dir / f"{fname_base_cm}.csv"
            pd.DataFrame(cm).to_csv(csv_cm, index=False, header=[f"pred_{i}" for i in range(n_classes)])
            print(f"[ARTIFACT] Saved confusion counts CSV: {csv_cm}")

        return

