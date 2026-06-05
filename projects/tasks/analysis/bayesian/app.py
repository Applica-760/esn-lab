import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import optuna

from projects.utils.app_init import build_param_grid
from projects.utils.eval.confusion import compute_confusion_matrix
from projects.utils.eval.metrics import compute_macro_f1_from_cm
from projects.utils.prediction import is_valid_result_file, load_pred_results
from projects.utils.weights import build_param_str


def judge_sample(predictions: np.ndarray, class_weights: np.ndarray, margin_threshold: float) -> int:
    """加重スコア＋margin棄権による単サンプル判定。全棄権時は全フレームにフォールバック"""
    sorted_scores = np.sort(predictions, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    valid_mask = margins >= margin_threshold
    valid_preds = predictions[valid_mask] if valid_mask.any() else predictions
    return int(np.argmax((valid_preds * class_weights).mean(axis=0)))


def evaluate_samples(
    samples: list, class_weights: np.ndarray, margin_threshold: float, n_classes: int
) -> float:
    """1 (group, fold) のサンプルリストに対する macro F1"""
    preds = [judge_sample(s[0], class_weights, margin_threshold) for s in samples]
    trues = [s[1] for s in samples]
    cm = compute_confusion_matrix(trues, preds, n_classes)
    return compute_macro_f1_from_cm(cm)


def make_objective(samples: list, n_classes: int):
    def objective(trial: optuna.Trial) -> float:
        w0 = trial.suggest_float("w0", 0.1, 5.0)
        w1 = trial.suggest_float("w1", 0.1, 5.0)
        w2 = trial.suggest_float("w2", 0.1, 5.0)
        margin_threshold = trial.suggest_float("margin_threshold", 0.0, 0.8)
        return evaluate_samples(samples, np.array([w0, w1, w2]), margin_threshold, n_classes)

    return objective


def run_optimization(samples: list, n_classes: int, n_trials: int) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(samples, n_classes), n_trials=n_trials)
    return study


def save_fold_result(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"best_value": study.best_value, "best_params": study.best_params}
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(result, f, indent=2)
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)


def one_process(group, fold_index, samples, n_classes, n_trials, output_dir):
    """単一 (group, fold) の Optuna 最適化（ProcessPoolExecutor から呼び出し）"""
    if not samples:
        print(f"  skip (no data): group={group} fold={fold_index}")
        return None

    study = run_optimization(samples, n_classes, n_trials)
    save_fold_result(study, Path(output_dir) / group / f"fold_{fold_index}")
    print(f"  done: group={group} fold={fold_index} | macro F1={study.best_value:.4f} | {study.best_params}")
    return {
        "group": group,
        "fold_index": fold_index,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }


def compute_final_params(all_results: list) -> dict:
    """全 (group, fold) の best_params を平均・標準偏差で集約"""
    param_keys = list(all_results[0]["best_params"].keys())
    return {
        "n_results": len(all_results),
        "mean_best_value": float(np.mean([r["best_value"] for r in all_results])),
        "mean": {k: float(np.mean([r["best_params"][k] for r in all_results])) for k in param_keys},
        "std": {k: float(np.std([r["best_params"][k] for r in all_results])) for k in param_keys},
    }


def save_summary(all_results: list, final_params: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    param_keys = list(all_results[0]["best_params"].keys())
    with open(output_dir / "all_best_params.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "fold_index", "best_value"] + param_keys)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x["group"], x["fold_index"])):
            writer.writerow({"group": r["group"], "fold_index": r["fold_index"], "best_value": r["best_value"], **r["best_params"]})

    with open(output_dir / "final_params.json", "w") as f:
        json.dump(final_params, f, indent=2)

    print(f"  final params (mean): {final_params['mean']}")
    print(f"  final params (std):  {final_params['std']}")
    print(f"  mean macro F1:       {final_params['mean_best_value']:.4f}")


def main(cfg):
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)

    for params in param_grid:
        param_str = build_param_str(params)
        print(f"\n=== {param_str} ===")
        param_output_dir = cfg.output_dir / param_str

        all_results = []
        n_classes = None

        for group in cfg.groups:
            result_base = str(pred_result_dir / group / param_str / "train_results")
            if not is_valid_result_file(result_base):
                continue

            fold_map = defaultdict(list)
            for fold_data in load_pred_results(result_base):
                fi = fold_data["fold_index"]
                for s in fold_data["results"]:
                    preds = np.asarray(s["predictions"])
                    labels = np.asarray(s["labels"])
                    true_frames = np.argmax(labels, axis=1)
                    true_label = int(np.argmax(np.bincount(true_frames, minlength=preds.shape[1])))
                    fold_map[fi].append((preds, true_label))
                    if n_classes is None:
                        n_classes = preds.shape[1]

            jobs = [
                (group, fi, samples, n_classes, cfg.n_trials, param_output_dir)
                for fi, samples in sorted(fold_map.items())
            ]

            print(f"  group={group}")
            with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
                futures = [executor.submit(one_process, *job) for job in jobs]
                all_results += [f.result() for f in futures]

        all_results = [r for r in all_results if r is not None]
        if all_results:
            final_params = compute_final_params(all_results)
            save_summary(all_results, final_params, param_output_dir / "summary")
        else:
            print("  skip (no data)")

    print("\noptimization finished")
