"""
混同行列関連のユーティリティ（計算・I/O・可視化）
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_confusion_matrix(y_true: list, y_pred: list, n_classes: int) -> np.ndarray:
    """
    混同行列を計算
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def compute_cm_from_judgment_results(judgment_results: list, n_classes: int, 
                                      fold_index: int = None, group: str = None) -> np.ndarray:
    """
    判定結果リストから混同行列を計算
    """
    y_true = []
    y_pred = []

    for result in judgment_results:
        # フィルタリング
        if fold_index is not None and result["fold_index"] != fold_index:
            continue
        if group is not None and result["group"] != group:
            continue

        y_true.append(result["true_label"])
        y_pred.append(result["pred_label"])

    return compute_confusion_matrix(y_true, y_pred, n_classes)


def load_confusion_matrix_csv(csv_path: str) -> np.ndarray:
    """
    CSV形式の混同行列を読み込む
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダー行をスキップ
        cm = []
        for row in reader:
            cm.append([int(val) for val in row[1:]])  # 最初の列（ラベル）をスキップ
    return np.array(cm, dtype=int)


def save_confusion_matrix_csv(cm: np.ndarray, class_names: list, output_path: str,
                               class_order: list = None) -> None:
    """
    混同行列をCSV形式で保存
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CMを表示順に並び替え（class_namesは既に表示順なので並び替えない）
    if class_order is not None:
        cm = cm[np.ix_(class_order, class_order)]

    # CSV保存
    with open(str(output_path) + ".csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = [""] + [f"pred_{name}" for name in class_names]
        writer.writerow(header)
        for i, row in enumerate(cm):
            writer.writerow([f"true_{class_names[i]}"] + list(row))


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, output_path: str,
                          class_order: list = None) -> None:
    """
    混同行列を画像として保存（行ごとに正規化、0〜1の値、PNG/PDF両形式）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CMを表示順に並び替え（class_namesは既に表示順なので並び替えない）
    if class_order is not None:
        cm = cm[np.ix_(class_order, class_order)]
    
    # 行ごとに正規化
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # ラベルを回転
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 各セルに数値を表示（正規化値のみ）
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_normalized[i, j]:.2f}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    fontsize=12,
                    color="white" if cm_normalized[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(str(output_path) + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(str(output_path) + ".pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: list, title: str, 
                          output_path: str, class_order: list = None) -> None:
    """
    混同行列をCSVと画像(PNG/PDF)で保存（convenience関数）
    """
    save_confusion_matrix_csv(cm, class_names, output_path, class_order)
    plot_confusion_matrix(cm, class_names, title, output_path, class_order)
