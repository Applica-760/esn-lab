from io import BytesIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def draw_source_image(axes, source_image):
    height, width = source_image.shape
    # 半画素分の境界を指定して列中心を整数の時刻に合わせる。
    axes.imshow(
        source_image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        aspect="auto",
        interpolation="nearest",
    )
    axes.set(title="Brightness distribution", ylabel="Image row")


def draw_node_output(axes, values, node_name):
    if values.ndim == 2:
        for index, (label, color) in enumerate(
            (("foraging", "tab:blue"), ("rumination", "tab:orange"))
        ):
            axes.plot(
                np.arange(len(values)), values[:, index], linewidth=0.8, label=label, color=color
            )
        axes.legend(loc="upper right")
    else:
        axes.plot(np.arange(len(values)), values, linewidth=0.8)
    axes.set(
        xlim=(0, max(len(values) - 1, 1)),
        ylim=(-1.5, 2.5),
        xlabel="Time step",
        ylabel="Raw output",
        title=node_name,
    )


def render_node_png(values, node_name, source_image=None):
    """保存系列の全時刻を描画し、ファイルを作らずPNGを返す。"""
    figure = Figure(figsize=(10, 3 if source_image is None else 5), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    if source_image is None:
        axes = figure.subplots()
    else:
        image_axes, axes = figure.subplots(2, 1, sharex=True, height_ratios=(1, 2))
        draw_source_image(image_axes, source_image)
        for item in (image_axes, axes):
            item.grid(axis="x", color="gray", alpha=0.4, linewidth=0.5)
    draw_node_output(axes, values, node_name)
    if source_image is not None:
        axes.set(xlim=(-0.5, len(values) - 0.5), xlabel="Time index (image column)")
    output = BytesIO()
    canvas.print_png(output)
    return output.getvalue()
