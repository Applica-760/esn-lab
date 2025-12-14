from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_errorbar_and_save(
    xs: Sequence,
    means: Sequence[float],
    stds: Sequence[float],
    vary_param: str,
    metric: str,
    title: str | None,
    out_dir: Path,
    *,
    dpi: int = 150,
    ylim: tuple[float, float] | None = None,
    counts: Sequence[int] | None = None,
    fname_base: str | None = None,
) -> tuple[Path, Path]:
    """Plot an errorbar chart and save both PNG and CSV (aggregated table).

    Returns (png_path, csv_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = list(xs)
    means = list(means)
    stds = list(stds)
    counts = list(counts) if counts is not None else [None] * len(xs)

    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(xs)), 4))
    marker_style = dict(fmt='o', linestyle='none', markersize=6, markerfacecolor='C0', markeredgecolor='C0')
    ax.errorbar(xs, means, yerr=stds, capsize=4, elinewidth=1.0, **marker_style)
    ax.set_xlabel(vary_param)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {vary_param} (mean±std)")
    ax.grid(True, linestyle='--', alpha=0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()

    fname_base = fname_base or f"errorbar_{metric}_by_{vary_param}"
    png_path = out_dir / f"{fname_base}.png"
    fig.savefig(png_path, dpi=int(dpi))
    plt.close(fig)
    print(f"[ARTIFACT] Saved errorbar plot: {png_path}")

    # Aggregated table
    out_df = pd.DataFrame({vary_param: xs, f"{metric}_mean": means, f"{metric}_std": stds, "count": counts})
    csv_path = out_dir / f"{fname_base}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"[ARTIFACT] Saved summary CSV: {csv_path}")

    return png_path, csv_path


def plot_confusion_matrices_and_save(
    dfp: pd.DataFrame,
    xs: Iterable,
    vary_param: str,
    n_classes: int,
    out_dir: Path,
    *,
    dpi: int = 150,
    base_prefix: str = "confusion_",
) -> None:
    """Plot and save confusion matrices for each value in xs.

    - dfp: predictions DataFrame containing columns ["true_label", "pred_label", vary_param]
    - xs: values of vary_param to iterate over
    - n_classes: number of classes to expect
    - Saves three outputs per value: PNG, counts CSV, normalized CSV
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide display order and labels
    if int(n_classes) == 3:
        perm = [1, 2, 0]
        tick_labels = ["foraging", "rumination", "other"]
    else:
        perm = list(range(int(n_classes)))
        tick_labels = [str(i) for i in perm]

    for v in xs:
        series = dfp[vary_param]
        if pd.api.types.is_numeric_dtype(series) and isinstance(v, (int, float)):
            tol = 1e-9
            df_v = dfp[(series - float(v)).abs() < tol]
        else:
            df_v = dfp[series.astype(str) == str(v)]

        if df_v.empty:
            print(f"[WARN] No prediction rows for {vary_param}={v}. Skipping confusion plot.")
            continue

        # Build counts matrix
        n = int(n_classes)
        cm = np.zeros((n, n), dtype=int)
        for _, rowp in df_v.iterrows():
            t = int(rowp["true_label"])
            p = int(rowp["pred_label"])
            if 0 <= t < n and 0 <= p < n:
                cm[t, p] += 1

        # Reorder for display
        cm_disp = cm[np.ix_(perm, perm)]

        # Row-normalize for plotting (0..1)
        cm_norm = cm_disp.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        cm_norm = cm_norm / row_sums

        # Plot
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('fraction', rotation=270, labelpad=12)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f"Confusion matrix: {vary_param}={v} | normalized (row)")
        ax.set_xticks(range(len(perm)))
        ax.set_yticks(range(len(perm)))
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        vmax = cm_norm.max() if cm_norm.size else 0
        thresh = vmax / 2 if vmax > 0 else 0.5
        for i in range(len(perm)):
            for j in range(len(perm)):
                frac = cm_norm[i, j]
                text = f"{frac:.2f}"
                ax.text(j, i, text, ha='center', va='center',
                        color='white' if frac > thresh else 'black')

        fname_base_cm = f"{base_prefix}{vary_param}-{v}"
        png_cm = out_dir / f"{fname_base_cm}.png"
        fig.savefig(png_cm, dpi=int(dpi), bbox_inches='tight')
        plt.close(fig)
        print(f"[ARTIFACT] Saved confusion matrix: {png_cm}")

        # Save counts CSV in the same display order
        csv_cm = out_dir / f"{fname_base_cm}.csv"
        pd.DataFrame(cm_disp).to_csv(csv_cm, index=False, header=[f"pred_{i}" for i in perm])
        print(f"[ARTIFACT] Saved confusion counts CSV: {csv_cm}")

        # Save normalized confusion matrix
        csv_cm_norm = out_dir / f"{fname_base_cm}_normalized.csv"
        pd.DataFrame(cm_norm).to_csv(csv_cm_norm, index=False, header=[f"pred_{i}" for i in perm])
        print(f"[ARTIFACT] Saved normalized confusion CSV: {csv_cm_norm}")
