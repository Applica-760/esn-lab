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
    value_range: tuple = (0, 1)
) -> None:
    """
    汎用ヒストグラムプロット関数
    """
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, range=value_range, color=color, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xlim(value_range)
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




