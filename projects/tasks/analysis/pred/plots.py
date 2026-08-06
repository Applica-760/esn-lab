from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_temporal_accuracy(accuracy, n_bins, class_names, class_order, output_dir):
    x_values = np.linspace(0, 100, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, class_index in enumerate(class_order):
        ax.plot(x_values, accuracy[class_index], label=class_names[index])
    ax.set_xlabel("Relative position (%)")
    ax.set_ylabel("Frame accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Temporal accuracy by true class")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "temporal_accuracy.png")


def plot_score_trajectory(means, standard_deviations, n_bins, class_names, class_order, output_dir):
    n_classes = len(class_names)
    x_values = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for index, true_class in enumerate(class_order):
        ax = axes[index]
        ax.set_title(f"true = {class_names[index]}")
        for score_index, score_class in enumerate(class_order):
            score_means = means[true_class, score_class]
            score_stds = standard_deviations[true_class, score_class]
            color = f"C{score_index}"
            ax.plot(x_values, score_means, label=class_names[score_index], color=color)
            ax.fill_between(
                x_values,
                score_means - score_stds,
                score_means + score_stds,
                alpha=0.2,
                color=color,
            )
        ax.set_xlabel("Relative position (%)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean score")
    fig.suptitle("Score trajectory by true class")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "score_trajectory.png")


def plot_stability(instabilities, class_names, class_order, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
    for index, class_index in enumerate(class_order):
        ax = axes[index]
        values = instabilities[class_index]
        ax.hist(values, bins=20, range=(0, 1), edgecolor="white")
        ax.set_title(f"true = {class_names[index]} (n={len(values)})")
        ax.set_xlabel("Argmax change rate")
        ax.set_ylabel("Count")
    fig.suptitle("Within-sample stability by true class")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "stability.png")


def plot_argmax_heatmap(heatmap, boundaries, class_names, class_order, output_dir):
    n_classes = len(class_names)
    colors = [f"C{class_order.index(class_index)}" for class_index in range(n_classes)]
    color_map = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5), color_map.N)

    fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap) // 8)))
    image = ax.imshow(
        heatmap,
        aspect="auto",
        cmap=color_map,
        norm=norm,
        extent=[0, 100, len(heatmap), 0],
        interpolation="nearest",
    )
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.0)

    color_bar = fig.colorbar(image, ax=ax, ticks=range(n_classes))
    color_bar.ax.set_yticklabels(
        [class_names[class_order.index(class_index)] for class_index in range(n_classes)]
    )
    ax.set_xlabel("Relative position (%)")
    ax.set_ylabel("Sample (sorted by true class)")
    ax.set_title("Argmax heatmap")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "argmax_heatmap.png")


def plot_score_margin(margins, class_names, class_order, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for index, class_index in enumerate(class_order):
        ax = axes[index]
        values = margins[class_index]
        ax.hist(values, bins=20, edgecolor="white")
        ax.set_title(f"true = {class_names[index]} (n={len(values)})")
        ax.set_xlabel("Mean margin (max − 2nd max)")
        ax.set_ylabel("Count")
    fig.suptitle("Score margin distribution by true class")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "score_margin.png")


def plot_temporal_accuracy_cells(cell_data, n_bins, class_names, class_order, output_dir):
    n_classes = len(class_names)
    x_values = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(
        n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True
    )

    for row_index, true_class in enumerate(class_order):
        for column_index, pred_class in enumerate(class_order):
            ax = axes[row_index, column_index]
            values = cell_data[(true_class, pred_class)]
            ax.set_title(
                f"true={class_names[row_index]}, pred={class_names[column_index]} "
                f"(n={values['count']})",
                fontsize=8,
            )
            if column_index == 0:
                ax.set_ylabel("Frame accuracy")
            if row_index == n_classes - 1:
                ax.set_xlabel("Relative position (%)", fontsize=7)
            ax.set_ylim(0, 1)
            if not values["count"]:
                continue
            ax.plot(x_values, values["means"], color="steelblue")
            ax.fill_between(
                x_values,
                values["means"] - values["stds"],
                values["means"] + values["stds"],
                alpha=0.2,
                color="steelblue",
            )

    fig.suptitle("Temporal accuracy (true × pred class)")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "temporal_accuracy_3x3.png")


def plot_score_trajectory_cells(cell_data, n_bins, class_names, class_order, output_dir):
    n_classes = len(class_names)
    x_values = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(
        n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True
    )

    for row_index, true_class in enumerate(class_order):
        for column_index, pred_class in enumerate(class_order):
            ax = axes[row_index, column_index]
            values = cell_data[(true_class, pred_class)]
            ax.set_title(
                f"true={class_names[row_index]}, pred={class_names[column_index]} "
                f"(n={values['count']})",
                fontsize=8,
            )
            if column_index == 0:
                ax.set_ylabel("Mean score")
            if row_index == n_classes - 1:
                ax.set_xlabel("Relative position (%)", fontsize=7)
            if not values["count"]:
                continue

            for score_index, score_class in enumerate(class_order):
                means = values["means"][score_class]
                standard_deviations = values["stds"][score_class]
                color = f"C{score_index}"
                ax.plot(x_values, means, label=class_names[score_index], color=color)
                ax.fill_between(
                    x_values,
                    means - standard_deviations,
                    means + standard_deviations,
                    alpha=0.2,
                    color=color,
                )
            ax.legend(fontsize=6)

    fig.suptitle("Score trajectory (true × pred class)")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "score_trajectory_3x3.png")


def plot_stability_cells(cell_data, class_names, class_order, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True
    )

    for row_index, true_class in enumerate(class_order):
        for column_index, pred_class in enumerate(class_order):
            ax = axes[row_index, column_index]
            values = cell_data[(true_class, pred_class)]
            ax.set_title(
                f"true={class_names[row_index]}, pred={class_names[column_index]} "
                f"(n={len(values)})",
                fontsize=8,
            )
            if column_index == 0:
                ax.set_ylabel("Count")
            if row_index == n_classes - 1:
                ax.set_xlabel("Argmax change rate", fontsize=7)
            if values:
                ax.hist(values, bins=20, range=(0, 1), edgecolor="white")

    fig.suptitle("Within-sample stability (true × pred class)")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "stability_3x3.png")


def plot_score_margin_cells(cell_data, class_names, class_order, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True
    )

    for row_index, true_class in enumerate(class_order):
        for column_index, pred_class in enumerate(class_order):
            ax = axes[row_index, column_index]
            values = cell_data[(true_class, pred_class)]
            ax.set_title(
                f"true={class_names[row_index]}, pred={class_names[column_index]} "
                f"(n={len(values)})",
                fontsize=8,
            )
            if column_index == 0:
                ax.set_ylabel("Count")
            if row_index == n_classes - 1:
                ax.set_xlabel("Mean margin (max − 2nd max)", fontsize=7)
            if values:
                ax.hist(values, bins=20, edgecolor="white")

    fig.suptitle("Score margin distribution (true × pred class)")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "score_margin_3x3.png")


def plot_fold_temporal_accuracy(fold_accuracies, n_bins, class_names, class_order, output_dir):
    x_values = np.linspace(0, 100, n_bins)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for index, class_index in enumerate(class_order):
        ax = axes[index]
        ax.set_title(f"true = {class_names[index]}")
        fold_curves = np.array([accuracy[class_index] for accuracy in fold_accuracies])
        for curve in fold_curves:
            ax.plot(x_values, curve, alpha=0.3, linewidth=0.8, color="steelblue")
        ax.plot(
            x_values,
            np.nanmean(fold_curves, axis=0),
            linewidth=2.0,
            color="steelblue",
            label="mean",
        )
        ax.set_xlabel("Relative position (%)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frame accuracy")
    fig.suptitle("Temporal accuracy across folds")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "temporal_accuracy_folds.png")


def plot_fold_stability(fold_indices, fold_stabilities, class_names, class_order, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for index, class_index in enumerate(class_order):
        ax = axes[index]
        ax.set_title(f"true = {class_names[index]}")
        positions = []
        plot_data = []
        for fold_position, _ in enumerate(fold_indices):
            values = fold_stabilities[fold_position][class_index]
            if len(values) > 1:
                positions.append(fold_position + 1)
                plot_data.append(values)
        if plot_data:
            ax.violinplot(plot_data, positions=positions, showmedians=True)
        ax.set_xticks(range(1, len(fold_indices) + 1))
        ax.set_xticklabels([str(fold_index) for fold_index in fold_indices], fontsize=7)
        ax.set_xlabel("Fold")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Argmax change rate")
    fig.suptitle("Stability distribution across folds")
    fig.tight_layout()
    _save_figure(fig, Path(output_dir) / "stability_folds.png")
