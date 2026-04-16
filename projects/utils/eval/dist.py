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
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # ヒストグラム描画
    n, bins_edges, patches = ax1.hist(
        values, bins=bins, range=value_range, 
        color=color, edgecolor='black', alpha=0.7,
        label=f'n={len(values)}' if show_count else None
    )
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(value_range)
    
    if show_count:
        ax1.legend(loc='upper left')
    
    # 累積分布の追加
    if show_cumulative:
        ax2 = ax1.twinx()
        
        # 累積分布を計算（0-1にスケール）
        cumulative = np.cumsum(n) / len(values)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        
        ax2.plot(bin_centers, cumulative, color='red', linewidth=2, marker='o', markersize=3)
        ax2.set_ylabel('Cumulative Distribution', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 1)
    
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
) -> None:
    """
    N×N の混同分布ヒストグラムを1枚の figure に描画する。
    行 = true_label、列 = predicted_class。

    data: {true_label_idx: {pred_class_idx: [ratio, ...]}}
    """
    N = len(class_names)
    fig, axes = plt.subplots(N, N, figsize=(4 * N, 3 * N))

    for row_i, row_name in enumerate(class_names):
        true_idx = class_order[row_i]
        for col_j, col_name in enumerate(class_names):
            pred_idx = class_order[col_j]
            ax = axes[row_i][col_j]
            ratios = data.get(true_idx, {}).get(pred_idx, [])

            color = colors[col_name]
            if ratios:
                n, bins_edges, _ = ax.hist(
                    ratios, bins=bins, range=(0, 1),
                    color=color, edgecolor='black', alpha=0.7
                )
                if show_cumulative:
                    ax2 = ax.twinx()
                    cumulative = np.cumsum(n) / len(ratios)
                    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
                    ax2.plot(bin_centers, cumulative, color='red', linewidth=1.5, marker='o', markersize=2)
                    ax2.set_ylim(0, 1)
                    ax2.tick_params(axis='y', labelsize=6)

            label = f"n={len(ratios)}" if show_count else ""
            ax.set_title(f"true={row_name} / pred={col_name}\n{label}", fontsize=8)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=6)
            if row_i == N - 1:
                ax.set_xlabel("ratio", fontsize=7)
            if col_j == 0:
                ax.set_ylabel("Freq", fontsize=7)

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


def compute_true_class_output(predictions, labels) -> np.ndarray:
    """
    各時刻での正解クラスの出力値を計算
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 真のラベル（多数決で決定）
    true_frames = np.argmax(labels, axis=1)
    true_label = int(np.argmax(np.bincount(true_frames)))
    
    # 各時刻の正解クラスの出力値を取得
    return predictions[:, true_label]




