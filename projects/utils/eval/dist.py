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


def plot_ratio_histogram(ratios: list, output_path: Path, bins: int, color: str) -> None:
    """
    横軸：count_true_pred_ratioで求めた真のラベルに該当する予測回数の%
    縦軸：その度数
    ヒストグラムなど適切なものを用いて分布がわかりやすいようにプロットする
    """
    plt.figure(figsize=(8, 6))
    plt.hist(ratios, bins=bins, range=(0, 1), color=color, edgecolor='black', alpha=0.7)
    plt.xlabel('True Label Prediction Ratio')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

