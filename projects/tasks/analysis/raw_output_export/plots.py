from io import BytesIO

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from projects.tasks.analysis.raw_output_viewer.plots import draw_node_output, draw_source_image


def render_group_png(sample_id, group, parameter, records, source_image, source_info):
    rows = (len(records) + 2) // 3
    figure = Figure(figsize=(18, 2.8 + 2.9 * rows), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    grid = figure.add_gridspec(rows + 1, 3, height_ratios=[1, *([2] * rows)])
    image_axis = figure.add_subplot(grid[0, :])
    output_axes = [
        figure.add_subplot(grid[1 + index // 3, index % 3], sharex=image_axis)
        for index in range(len(records))
    ]
    axes = [image_axis, *output_axes]
    counts = []
    for mode in ("train", "test"):
        selected = [r for r in records if r["mode"] == mode]
        counts.append(
            f"{mode}: errors={sum(not r['is_correct'] for r in selected)}/{len(selected)}"
            if selected
            else f"{mode}: no evaluations"
        )
    figure.suptitle(
        f"ID {sample_id} | true={records[0]['true_label']} | Group {group}\n"
        f"{parameter}\n" + " | ".join(counts),
        fontsize=12,
    )
    if source_image is not None:
        draw_source_image(axes[0], source_image)
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Source image: {source_info['status']}",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_ylabel("Image unavailable")
        axes[0].set_yticks([])
    for axis, record in zip(axes[1:], records):
        votes = record["votes"]
        pred = votes["pred_label"]
        result = "correct" if record["is_correct"] else "WRONG"
        judgment = record["judgment"]
        if votes["matches_saved_judgment"] is True:
            saved = "saved: match"
        elif votes["matches_saved_judgment"] is False:
            saved = f"saved: MISMATCH (pred={judgment['pred_label']})"
        else:
            saved = f"saved: {judgment['status']}"
        warning = "\nOUTPUT OUTSIDE [-1.5, 2.5]" if record["out_of_range"] else ""
        title = (
            f"Fold {record['fold_index']} | {record['mode'].upper()} | "
            f"recomputed: {result}\n"
            f"pred={pred} | vote={votes['ratios'][pred]:.1%}\n"
            f"{saved}{warning}"
        )
        draw_node_output(axis, record["predictions"], title)
        axis.set_xlabel("Time index (image column)")
        axis.title.set_fontsize(10)
        axis.title.set_color("black" if record["is_correct"] else "firebrick")
    for axis in axes:
        axis.grid(axis="x", color="gray", alpha=0.4, linewidth=0.5)
    axes[-1].set(xlim=(-0.5, records[0]["length"] - 0.5), xlabel="Time index (image column)")
    output = BytesIO()
    canvas.print_png(output)
    return output.getvalue()
