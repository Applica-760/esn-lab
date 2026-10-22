from pathlib import Path

import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle


def build_heatmap(sample_ids, conditions, cells, labels, title, combined=False):
    values = np.full((len(sample_ids), len(conditions)), np.nan)
    markers = []
    test_cells = []
    for y, sample_id in enumerate(sample_ids):
        for x, (group, fold) in enumerate(conditions):
            row = cells[(sample_id, group, fold)]
            status = row["status"]
            if (
                combined
                and row["mode"] == "test"
                and status not in ("missing_sample", "missing_file", "invalid_file")
            ):
                test_cells.append((x, y))
            if status.startswith("invalid_") or status == "judgment_mismatch":
                markers.append((x, y, "!"))
            elif row["judgment_status"] == "invalid_data":
                markers.append((x, y, "!"))
            elif status == "available":
                values[y, x] = 100 * row["true_ratio"]
                if not row["is_correct"]:
                    markers.append((x, y, "X"))
                if row["judgment_status"] != "available":
                    markers.append((x + 0.28, y - 0.22, "?"))
            elif status == "missing_file":
                markers.append((x, y, "-"))

    figure = Figure(
        figsize=(max(9, 0.25 * len(conditions) + 3), max(4, 0.24 * len(sample_ids) + 3)),
        layout="constrained",
    )
    axis = figure.subplots()
    cmap = colormaps["RdYlGn"].copy()
    cmap.set_bad("#dddddd")
    image = axis.imshow(values, cmap=cmap, norm=Normalize(0, 100), aspect="auto")
    axis.set_xticks(range(len(conditions)), [f"{g}/{f}" for g, f in conditions], rotation=90)
    axis.set_yticks(
        range(len(sample_ids)),
        [f"{sample_id} [{labels[sample_id] or 'unknown'}]" for sample_id in sample_ids],
    )
    axis.tick_params(labelsize=8)
    axis.set(xlabel="Group / Fold", ylabel="ID [true label]", title=title)
    for index in range(1, len(conditions)):
        if conditions[index - 1][0] != conditions[index][0]:
            axis.axvline(
                index - 0.5,
                color="white",
                linewidth=1.2,
                zorder=2.5,
                path_effects=[
                    patheffects.Stroke(linewidth=3.2, foreground="black"),
                    patheffects.Normal(),
                ],
            )
        else:
            axis.axvline(index - 0.5, color="white", linewidth=0.5, alpha=0.5, zorder=2.5)
    for x, y, marker in markers:
        incorrect = marker == "X"
        axis.text(
            x,
            y,
            marker,
            ha="center",
            va="center",
            zorder=4,
            fontsize=9 if incorrect else 7,
            color="white" if incorrect else "black",
            fontweight="bold" if incorrect else "normal",
        )
    for x, y in test_cells:
        axis.add_patch(
            Rectangle(
                (x - 0.44, y - 0.44),
                0.88,
                0.88,
                fill=False,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                path_effects=[
                    patheffects.Stroke(linewidth=2.6, foreground="black"),
                    patheffects.Normal(),
                ],
            )
        )
    figure.colorbar(image, ax=axis, label="Votes for true label (%)", ticks=[0, 25, 50, 75, 100])
    legend = [Patch(facecolor="#dddddd", label="No usable prediction")]
    if combined:
        legend.extend(
            [
                Patch(
                    facecolor="#888888",
                    edgecolor="white",
                    linewidth=0.8,
                    path_effects=[
                        patheffects.Stroke(linewidth=2.6, foreground="black"),
                        patheffects.Normal(),
                    ],
                    label="Test: black/white outline",
                ),
                Patch(facecolor="#888888", edgecolor="none", label="Train: no outline"),
            ]
        )
    axis.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        frameon=False,
        fontsize=8,
        ncol=len(legend),
    )
    figure.supxlabel(
        "X: incorrect   !: invalid / saved judgment mismatch   -: prediction file missing\n"
        "?: saved judgment unavailable (recomputed votes shown); ties select foraging",
        fontsize=9,
    )
    return figure


def _combined_cells(rows):
    grouped = {}
    for row in rows:
        key = (row["id"], row["group"], row["fold_index"])
        grouped.setdefault(key, []).append(row)
    cells = {}
    for key, pair in grouped.items():
        present = [
            r for r in pair if r["status"] not in ("missing_sample", "missing_file", "invalid_file")
        ]
        if len(present) > 1:
            raise ValueError(f"train/test overlap in combined heatmap: {key}")
        invalid = next((r for r in pair if r["status"] == "invalid_file"), None)
        missing = next((r for r in pair if r["status"] == "missing_file"), pair[0])
        cells[key] = invalid or (present[0] if present else missing)
    return cells


def render_heatmaps(rows, output_dir, groups, folds, parameters, modes, rows_per_page=50):
    if type(rows_per_page) is not int or rows_per_page < 1:
        raise ValueError("rows_per_page must be a positive integer")
    output_dir = Path(output_dir).resolve()
    labels = {row["id"]: row["true_label"] for row in rows}
    sample_ids = sorted(labels, key=lambda sample_id: (labels[sample_id], sample_id))
    conditions = [(group, fold) for group in groups for fold in sorted(folds)]
    pages = (len(sample_ids) + rows_per_page - 1) // rows_per_page
    paths = []
    for parameter in parameters:
        selected = [row for row in rows if row["parameter"] == parameter and row["mode"] in modes]
        views = list(modes)
        combined_cells = None
        if set(modes) == {"train", "test"}:
            combined_cells = _combined_cells(selected)
            views.append("combined")
        directory = (output_dir / parameter).resolve()
        if not directory.is_relative_to(output_dir):
            raise ValueError("plot directory must stay inside output directory")
        directory.mkdir(parents=True, exist_ok=True)
        for mode in views:
            cells = (
                combined_cells
                if mode == "combined"
                else {
                    (row["id"], row["group"], row["fold_index"]): row
                    for row in selected
                    if row["mode"] == mode
                }
            )
            for page in range(pages):
                ids = sample_ids[page * rows_per_page : (page + 1) * rows_per_page]
                figure = build_heatmap(
                    ids,
                    conditions,
                    cells,
                    labels,
                    f"{parameter} | {mode} | page {page + 1}/{pages}",
                    combined=mode == "combined",
                )
                path = directory / f"{mode}_{page + 1:03d}.png"
                if path.is_symlink():
                    raise ValueError("output PNG must not be a symbolic link")
                figure.savefig(path, dpi=150)
                figure.clear()
                paths.append(path)
    return paths
