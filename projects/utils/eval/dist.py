import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def count_true_pred_ratio(predictions, labels) -> float:
    """
    judge_sample_by_majority_voteを参考に，
    「フレームごとの予測ラベルのうち，真のラベルに該当するものが全体の時系列長の何%か」を取得して，小数点以下2桁で丸めて返す
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # フレーム単位のラベルに変換（argmax）
    pred_frames = np.argmax(predictions, axis=1)
    true_frames = np.argmax(labels, axis=1)

    # 真のラベル（多数決）を取得
    true_label = int(np.argmax(np.bincount(true_frames)))

    # 真のラベルと一致するフレーム数をカウント
    match_count = np.sum(pred_frames == true_label)
    total_frames = len(pred_frames)

    # %を算出して小数点以下2桁で丸める
    ratio = round(match_count / total_frames, 2)
    return ratio


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




