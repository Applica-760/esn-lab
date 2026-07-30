import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from projects.tasks.analysis.margin.common import CLASS_ORDER, collect_samples, margin_rows


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


def main(cfg):
    samples_by_param = collect_samples(Path(cfg.pred_result_dir), cfg.groups, cfg.warmup_ratio)
    for param_name, samples in samples_by_param.items():
        rows = margin_rows(samples)
        output_dir = Path(cfg.output_dir) / param_name
        write_csv(
            output_dir / "margin_by_sample.csv",
            rows,
            ["group", "fold_index", "id", "true_label", "margin", "frames_after_warmup"],
        )
        write_csv(
            output_dir / "margin_summary.csv",
            summarize_rows(rows),
            ["true_label", "count", "mean", "std", "median", "min", "max"],
        )
        plot_distribution(
            rows, output_dir / "margin_distribution.png", cfg.bins, tuple(cfg.x_range)
        )
        print(f"done: {param_name}")

    print("margin analysis is finished")
