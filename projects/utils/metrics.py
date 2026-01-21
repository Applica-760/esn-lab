"""
評価指標関連のユーティリティ（計算・可視化）
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict



def compute_accuracy_from_cm(cm: np.ndarray) -> float:
    """
    混同行列からAccuracyを計算
    """
    total = cm.sum()
    if total == 0:
        return 0.0
    return np.trace(cm) / total


def compute_macro_f1_from_cm(cm: np.ndarray) -> float:
    """
    混同行列からMacro F1を計算
    """
    n_classes = cm.shape[0]
    f1_scores = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return np.mean(f1_scores)


def extract_param_value(param_name: str, key: str) -> float:
    """
    パラメータ名から指定したキーの値を抽出
    例: extract_param_value("Nx7_dens0.5_inscl0.001_rho0.9", "Nx") → 7.0
    """
    pattern = rf"{key}([\d.]+)"
    match = re.search(pattern, param_name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Key '{key}' not found in param_name '{param_name}'")


def plot_metric_by_param(
    param_values: list,
    metric_values: list,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    ylim: list = None
) -> None:
    """
    横軸: パラメータ値（Nx等）
    縦軸: 性能指標（平均±標準偏差）
    同じパラメータ値を持つ結果を集約してエラーバー付きプロット
    """
    # パラメータ値ごとにmetric_valuesを集約
    grouped = defaultdict(list)
    for pv, mv in zip(param_values, metric_values):
        grouped[pv].append(mv)

    # ソートしてプロット用データを作成
    sorted_keys = sorted(grouped.keys())
    means = []
    stds = []
    for k in sorted_keys:
        values = grouped[k]
        means.append(np.mean(values))
        stds.append(np.std(values))

    # プロット
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(sorted_keys, means, yerr=stds, fmt='o-', capsize=5, capthick=2, markersize=8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Y軸の範囲を設定
    if ylim is None:
        ylim = [0, 1]
    ax.set_ylim(ylim[0], ylim[1])

    fig.tight_layout()
    plt.savefig(output_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path + ".pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_performance_summary(param_dirs, output_dir, param_key="Nx", ylim=None, mode=None):
    """
    全パラメータのtotal_{mode}.csvを読み込み、
    指定パラメータ（デフォルトはNx）別にAccuracyとMacro F1の平均・標準偏差をプロット
    
    Args:
        param_dirs: パラメータディレクトリのリスト
        output_dir: 出力ディレクトリ
        param_key: 横軸に使用するパラメータキー
        ylim: Y軸の範囲
        mode: train/test（ファイル名に含める）
    """
    from projects.utils.confusion import load_confusion_matrix_csv
    
    # mode suffix
    mode_suffix = f"_{mode}" if mode else ""
    
    param_values = []
    accuracies = []
    macro_f1s = []

    for param_dir in param_dirs:
        param_name = param_dir.name
        total_csv_path = output_dir / param_name / f"total{mode_suffix}.csv"

        if not total_csv_path.exists():
            print(f"skipped (total.csv not found): {param_name}")
            continue

        # 混同行列を読み込み
        cm = load_confusion_matrix_csv(str(total_csv_path))

        # 性能指標を計算
        accuracy = compute_accuracy_from_cm(cm)
        macro_f1 = compute_macro_f1_from_cm(cm)

        # パラメータ値を抽出
        try:
            pv = extract_param_value(param_name, param_key)
        except ValueError as e:
            print(f"skipped ({e}): {param_name}")
            continue

        param_values.append(pv)
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)

    if not param_values:
        print("No data to plot")
        return

    # Accuracyプロット
    plot_metric_by_param(
        param_values, accuracies,
        xlabel=param_key, ylabel="Accuracy",
        title=f"Accuracy by {param_key}" + (f" ({mode})" if mode else ""),
        output_path=str(output_dir / f"accuracy_by_{param_key}{mode_suffix}"),
        ylim=ylim
    )
    print(f"Saved: accuracy_by_{param_key}{mode_suffix}.png/pdf")

    # Macro F1プロット
    plot_metric_by_param(
        param_values, macro_f1s,
        xlabel=param_key, ylabel="Macro F1",
        title=f"Macro F1 by {param_key}" + (f" ({mode})" if mode else ""),
        output_path=str(output_dir / f"macro_f1_by_{param_key}{mode_suffix}"),
        ylim=ylim
    )
    print(f"Saved: macro_f1_by_{param_key}{mode_suffix}.png/pdf")
