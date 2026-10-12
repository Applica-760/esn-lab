from io import BytesIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def render_node_png(values, node_name):
    """保存系列の全時刻を描画し、ファイルを作らずPNGを返す。"""
    figure = Figure(figsize=(10, 3), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
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
    output = BytesIO()
    canvas.print_png(output)
    return output.getvalue()
