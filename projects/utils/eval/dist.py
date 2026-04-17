import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def count_all_class_ratios(predictions, labels) -> tuple:
    """
    各クラスへの予測フレーム割合をN個返す。
    Returns: (ratios, true_label)
        ratios: list[float] — クラスjと予測されたフレームの割合 (j=0..N-1)
        true_label: int     — 多数決による真のラベル
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    pred_frames = np.argmax(predictions, axis=1)
    true_frames = np.argmax(labels, axis=1)

    true_label = int(np.argmax(np.bincount(true_frames)))
    n_classes = predictions.shape[1]

    ratios = [round(float(np.mean(pred_frames == j)), 2) for j in range(n_classes)]
    return ratios, true_label


def _draw_histogram_on_ax(
    ax,
    values,
    bins: int,
    color: str,
    xlabel: str,
    value_range: tuple,
    show_count: bool,
    show_cumulative: bool,
    label_fontsize: int = 10,
    tick_fontsize: int = 8,
    cumulative_linewidth: float = 2.0,
    cumulative_markersize: float = 3.0,
) -> None:
    """
    既存の ax にヒストグラムと累積カウントを描画する内部ヘルパー。
    figure の作成・保存は呼び出し側が行う。
    """
    n, bins_edges, _ = ax.hist(
        values, bins=bins, range=value_range,
        color=color, edgecolor='black', alpha=0.7,
        label=f'n={len(values)}' if show_count else None
    )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel('Frequency', fontsize=label_fontsize)
    ax.set_xlim(value_range)
    ax.tick_params(labelsize=tick_fontsize)

    if show_count:
        ax.legend(loc='upper left', fontsize=tick_fontsize)

    if show_cumulative:
        ax2 = ax.twinx()
        cumulative = np.cumsum(n)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        ax2.plot(bin_centers, cumulative, color='red',
                 linewidth=cumulative_linewidth, marker='o', markersize=cumulative_markersize)
        ax2.set_ylabel('Cumulative Count', fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.set_ylim(0, len(values))


def plot_histogram(
    values: list,
    output_path: Path,
    bins: int,
    color: str,
    xlabel: str = 'True Label Prediction Ratio',
    value_range: tuple = (0, 1),
    show_count: bool = True,
    show_cumulative: bool = False
) -> None:
    """
    汎用ヒストグラムプロット関数
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_histogram_on_ax(
        ax, values, bins, color, xlabel, value_range,
        show_count, show_cumulative
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_distribution(
    data: dict,
    class_names: list,
    class_order: list,
    output_path: Path,
    bins: int,
    colors: dict,
    show_count: bool = True,
    show_cumulative: bool = False,
    value_range: tuple = (0, 1),
    xlabel: str = "ratio",
    col_label: str = "argmax",
) -> None:
    """
    N×N の混同分布ヒストグラムを1枚の figure に描画する。
    行 = true_label、列 = predicted_class (またはノードインデックス)。

    data: {true_label_idx: {col_idx: [values, ...]}}
    col_label: タイトルの列ラベル (例: "argmax", "node")
    """
    N = len(class_names)
    fig, axes = plt.subplots(N, N, figsize=(4 * N, 3 * N))

    for row_i, row_name in enumerate(class_names):
        true_idx = class_order[row_i]
        for col_j, col_name in enumerate(class_names):
            pred_idx = class_order[col_j]
            ax = axes[row_i][col_j]
            values = data.get(true_idx, {}).get(pred_idx, [])

            if values:
                _draw_histogram_on_ax(
                    ax, values, bins, colors[col_name],
                    xlabel=xlabel, value_range=value_range,
                    show_count=False, show_cumulative=show_cumulative,
                    label_fontsize=7, tick_fontsize=6,
                    cumulative_linewidth=1.5, cumulative_markersize=2.0,
                )

            label = f"n={len(values)}" if show_count else ""
            ax.set_title(f"true={row_name} / {col_label}={col_name}\n{label}", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# ノード生値の指標計算関数
# =============================================================================

def compute_confidence(predictions) -> np.ndarray:
    """
    各時刻での確信度（最大出力値）を計算
    """
    predictions = np.array(predictions)
    return np.max(predictions, axis=1)


def compute_margin(predictions) -> np.ndarray:
    """
    各時刻でのマージン（1位と2位の出力値の差）を計算
    """
    predictions = np.array(predictions)
    sorted_preds = np.sort(predictions, axis=1)
    # 1位（最大）- 2位（2番目に大きい）
    return sorted_preds[:, -1] - sorted_preds[:, -2]


def compute_true_class_output(predictions, labels, node_idx=None) -> np.ndarray:
    """
    各時刻での指定ノードの出力値を計算。
    node_idx=None の場合は真のクラス（多数決）を自動判定する。
    """
    predictions = np.array(predictions)

    if node_idx is None:
        labels = np.array(labels)
        true_frames = np.argmax(labels, axis=1)
        node_idx = int(np.argmax(np.bincount(true_frames)))

    return predictions[:, node_idx]




