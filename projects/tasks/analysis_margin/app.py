import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from projects.utils.prediction import is_valid_result_file, load_pred_results

CLASS_NAMES = {0: "other", 1: "foraging", 2: "rumination"}
CLASS_ORDER = ["foraging", "rumination", "other"]
PREDICTED_CLASS_NAMES = ["foraging", "rumination"]


def summarize_sample(predictions, labels, warmup_ratio: float) -> tuple[str, float, int]:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(f"expected predictions shape (timesteps, 2), got {predictions.shape}")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(f"expected labels shape (timesteps, 3), got {labels.shape}")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not 0 <= warmup_ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

    warmup = int(len(predictions) * warmup_ratio)
    predictions = predictions[warmup:]
    labels = labels[warmup:]
    if len(predictions) == 0:
        raise ValueError("no frames remain after warmup")

    frame_classes = np.argmax(labels, axis=1)
    if not np.all(frame_classes == frame_classes[0]):
        raise ValueError("labels must be single-class over all timesteps")
    true_label = CLASS_NAMES[int(frame_classes[0])]
    margin = float(np.abs(predictions[:, 0] - predictions[:, 1]).mean())
    return true_label, margin, len(predictions)


def collect_samples(
    pred_result_dir: Path, groups: list[str], warmup_ratio: float
) -> dict[str, list[dict]]:
    samples_by_param = defaultdict(list)
    for group in groups:
        group_dir = pred_result_dir / group
        if not group_dir.exists():
            print(f"skip (group not found): {group_dir}")
            continue

        for param_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            result_base = param_dir / "test_results"
            if not is_valid_result_file(str(result_base)):
                print(f"skip (result not found): {result_base}")
                continue

            for fold_data in load_pred_results(str(result_base)):
                fold_index = fold_data["fold_index"]
                for sample in fold_data["results"]:
                    true_label, margin, frames = summarize_sample(
                        sample["predictions"], sample["labels"], warmup_ratio
                    )
                    warmup = int(len(sample["predictions"]) * warmup_ratio)
                    predictions = np.asarray(sample["predictions"])[warmup:]
                    pred_frames = np.argmax(predictions, axis=1)
                    pred_counts = np.bincount(
                        pred_frames, minlength=len(PREDICTED_CLASS_NAMES)
                    )
                    pred_label = PREDICTED_CLASS_NAMES[int(np.argmax(pred_counts))]
                    samples_by_param[param_dir.name].append(
                        {
                            "group": group,
                            "fold_index": fold_index,
                            "id": sample["id"],
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "margin": margin,
                            "frames_after_warmup": frames,
                            "predictions": predictions,
                        }
                    )
    return samples_by_param


def collect_rows(
    pred_result_dir: Path, groups: list[str], warmup_ratio: float
) -> dict[str, list[dict]]:
    rows_by_param = defaultdict(list)
    for param_name, samples in collect_samples(pred_result_dir, groups, warmup_ratio).items():
        for sample in samples:
            rows_by_param[param_name].append(
                {
                    key: value
                    for key, value in sample.items()
                    if key != "predictions"
                }
            )
    return rows_by_param


def summarize_rows(rows: list[dict]) -> list[dict]:
    summary = []
    for true_label in CLASS_ORDER:
        margins = np.array([row["margin"] for row in rows if row["true_label"] == true_label])
        summary.append(
            {
                "true_label": true_label,
                "count": len(margins),
                "mean": float(np.mean(margins)) if len(margins) else np.nan,
                "std": float(np.std(margins)) if len(margins) else np.nan,
                "median": float(np.median(margins)) if len(margins) else np.nan,
                "min": float(np.min(margins)) if len(margins) else np.nan,
                "max": float(np.max(margins)) if len(margins) else np.nan,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def histogram_bin_edges(x_range: tuple[float, float], bins: int) -> np.ndarray:
    return np.linspace(*x_range, bins + 1)


def plot_distribution(
    rows: list[dict], output_path: Path, bins: int, x_range: tuple[float, float]
) -> None:
    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(5 * len(CLASS_ORDER), 4), sharey=True)
    bin_edges = histogram_bin_edges(x_range, bins)
    for axis, true_label in zip(axes, CLASS_ORDER):
        margins = [row["margin"] for row in rows if row["true_label"] == true_label]
        axis.hist(margins, bins=bin_edges, edgecolor="white")
        axis.set_xlim(x_range)
        axis.set_title(f"true = {true_label} (n={len(margins)})")
        axis.set_xlabel("Mean margin |y_foraging - y_rumination|")
        axis.set_ylabel("Count")
    fig.suptitle("2-class ESN margin distribution by true behavior")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def temporal_bin_indices(timesteps: int, bins: int) -> np.ndarray:
    return np.minimum((np.arange(timesteps) / timesteps * bins).astype(int), bins - 1)


def plot_score_trajectory(
    samples: list[dict], output_path: Path, bins: int
) -> None:
    fig, axes = plt.subplots(
        len(CLASS_ORDER),
        len(PREDICTED_CLASS_NAMES),
        figsize=(5 * len(PREDICTED_CLASS_NAMES), 4 * len(CLASS_ORDER)),
        sharey=True,
    )
    x = np.linspace(0, 100, bins)

    for row_index, true_label in enumerate(CLASS_ORDER):
        for col_index, pred_label in enumerate(PREDICTED_CLASS_NAMES):
            axis = axes[row_index, col_index]
            cell = [
                sample
                for sample in samples
                if sample["true_label"] == true_label and sample["pred_label"] == pred_label
            ]
            axis.set_title(f"true={true_label}, pred={pred_label} (n={len(cell)})", fontsize=8)
            if col_index == 0:
                axis.set_ylabel("Mean output")
            if row_index == len(CLASS_ORDER) - 1:
                axis.set_xlabel("Relative position (%)", fontsize=7)
            if not cell:
                continue

            output_data = [[[] for _ in range(bins)] for _ in PREDICTED_CLASS_NAMES]
            for sample in cell:
                predictions = sample["predictions"]
                bin_indices = temporal_bin_indices(len(predictions), bins)
                for bin_index in range(bins):
                    mask = bin_indices == bin_index
                    if mask.any():
                        bin_means = predictions[mask].mean(axis=0)
                        for output_index in range(len(PREDICTED_CLASS_NAMES)):
                            output_data[output_index][bin_index].append(bin_means[output_index])

            for output_index, output_name in enumerate(PREDICTED_CLASS_NAMES):
                data_per_bin = output_data[output_index]
                means = np.array([np.mean(data) if data else np.nan for data in data_per_bin])
                stds = np.array([np.std(data) if len(data) > 1 else 0.0 for data in data_per_bin])
                axis.plot(x, means, label=output_name, color=f"C{output_index}")
                axis.fill_between(
                    x, means - stds, means + stds, alpha=0.2, color=f"C{output_index}"
                )
            axis.legend(fontsize=6)

    fig.suptitle("2-class ESN output trajectory (true × predicted class)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(cfg):
    samples_by_param = collect_samples(Path(cfg.pred_result_dir), cfg.groups, cfg.warmup_ratio)
    for param_name, samples in samples_by_param.items():
        rows = [
            {key: value for key, value in sample.items() if key != "predictions"}
            for sample in samples
        ]
        output_dir = Path(cfg.output_dir) / param_name
        write_csv(
            output_dir / "margin_by_sample.csv",
            rows,
            [
                "group",
                "fold_index",
                "id",
                "true_label",
                "pred_label",
                "margin",
                "frames_after_warmup",
            ],
        )
        write_csv(
            output_dir / "margin_summary.csv",
            summarize_rows(rows),
            ["true_label", "count", "mean", "std", "median", "min", "max"],
        )
        plot_distribution(
            rows,
            output_dir / "margin_distribution.png",
            cfg.bins,
            tuple(cfg.x_range),
        )
        plot_score_trajectory(
            samples,
            output_dir / "score_trajectory_3x2.png",
            cfg.trajectory_bins,
        )
        print(f"done: {param_name}")

    print("margin analysis is finished")
