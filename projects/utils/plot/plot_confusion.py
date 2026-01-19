import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, output_path: str,
                          class_order: list = None) -> None:
    """
    混同行列を画像として保存（行ごとに正規化、0〜1の値、PNG/PDF両形式）
    
    Args:
        cm: 混同行列
        class_names: クラス名のリスト（表示順）
        title: プロットのタイトル
        output_path: 保存先パス（拡張子なし）
        class_order: クラスの並び替え順序（Noneの場合は並び替えなし）
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
