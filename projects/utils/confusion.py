import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_confusion_matrix(y_true: list, y_pred: list, n_classes: int) -> np.ndarray:
    """
    混同行列を計算

    Args:
        y_true: 正解ラベルのリスト
        y_pred: 予測ラベルのリスト
        n_classes: クラス数

    Returns:
        混同行列 (n_classes x n_classes)
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def accumulate_confusion_matrices(cm_list: list) -> np.ndarray:
    """
    複数の混同行列を累積

    Args:
        cm_list: 混同行列のリスト

    Returns:
        累積された混同行列
    """
    return np.sum(cm_list, axis=0)


def reorder_confusion_matrix(cm: np.ndarray, class_order: list) -> np.ndarray:
    """
    混同行列の行と列を指定した順序で並び替える

    Args:
        cm: 混同行列 (n_classes x n_classes)
        class_order: 表示順に対応する元のインデックスリスト
                     例: [1, 2, 0] → 元の1を0番目、元の2を1番目、元の0を2番目に

    Returns:
        並び替えられた混同行列
    """
    cm_reordered = cm[np.ix_(class_order, class_order)]
    return cm_reordered


def save_confusion_matrix(cm: np.ndarray, class_names: list, title: str, output_path: str,
                          class_order: list = None) -> None:
    """
    混同行列をPNG, PDF, CSV形式で保存

    Args:
        cm: 混同行列
        class_names: クラス名リスト（表示順）
        title: 図のタイトル
        output_path: 出力パス（拡張子なし）
        class_order: 表示順に対応する元のインデックスリスト（Noneなら並び替えなし）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 並び替え
    if class_order is not None:
        cm = reorder_confusion_matrix(cm, class_order)

    # CSV保存
    _save_confusion_matrix_csv(cm, class_names, str(output_path) + ".csv")

    # PNG, PDF保存
    _plot_confusion_matrix(cm, class_names, title, str(output_path) + ".png")
    _plot_confusion_matrix(cm, class_names, title, str(output_path) + ".pdf")


def _save_confusion_matrix_csv(cm: np.ndarray, class_names: list, output_path: str) -> None:
    """
    混同行列をCSV形式で保存
    """
    import csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ヘッダー行（予測ラベル）
        header = [""] + [f"pred_{name}" for name in class_names]
        writer.writerow(header)
        # データ行（真のラベル）
        for i, row in enumerate(cm):
            writer.writerow([f"true_{class_names[i]}"] + list(row))


def _plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, output_path: str) -> None:
    """
    混同行列を画像として保存（行ごとに正規化、0〜1の値）
    """
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
