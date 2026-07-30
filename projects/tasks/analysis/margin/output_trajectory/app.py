from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from projects.tasks.analysis.margin.common import (
    CLASS_ORDER,
    PREDICTED_CLASS_NAMES,
    collect_samples,
)


def predicted_label(predictions: np.ndarray) -> str:
    frame_labels = np.argmax(predictions, axis=1)
    counts = np.bincount(frame_labels, minlength=len(PREDICTED_CLASS_NAMES))
    return PREDICTED_CLASS_NAMES[int(np.argmax(counts))]


def temporal_bin_indices(timesteps: int, bins: int) -> np.ndarray:
    return np.minimum((np.arange(timesteps) / timesteps * bins).astype(int), bins - 1)


def plot_score_trajectory(samples: list[dict], output_path: Path, bins: int) -> None:
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
                if sample["true_label"] == true_label
                and predicted_label(sample["predictions"]) == pred_label
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
    samples_by_param = collect_samples(
        Path(cfg.pred_result_dir),
        cfg.groups,
        cfg.warmup_ratio,
        getattr(cfg, "fold_indices", None),
    )
    for param_name, samples in samples_by_param.items():
        output_dir = Path(cfg.output_dir) / param_name
        plot_score_trajectory(samples, output_dir / "score_trajectory_3x2.png", cfg.trajectory_bins)
        if getattr(cfg, "separate_fold_output", False):
            for group in cfg.groups:
                group_samples = [sample for sample in samples if sample["group"] == group]
                for fold_index in sorted({sample["fold_index"] for sample in group_samples}):
                    fold_samples = [
                        sample for sample in group_samples if sample["fold_index"] == fold_index
                    ]
                    plot_score_trajectory(
                        fold_samples,
                        output_dir / group / f"fold_{fold_index}" / "score_trajectory_3x2.png",
                        cfg.trajectory_bins,
                    )
        print(f"done: {param_name}")

    print("margin output trajectory analysis is finished")
