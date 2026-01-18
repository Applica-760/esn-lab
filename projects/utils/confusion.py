import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from projects.utils.result_saver import load_eval_results
from projects.utils.vote import majority_vote_prediction, majority_vote_label


def compute_confusion_matrix(y_true: list, y_pred: list, n_classes: int) -> np.ndarray:
    """
    混同行列を計算
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def accumulate_confusion_matrices(cm_list: list) -> np.ndarray:
    """
    複数の混同行列を累積
    """
    return np.sum(cm_list, axis=0)


def load_confusion_matrix(csv_path: str) -> np.ndarray:
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


def save_confusion_matrix(cm: np.ndarray, class_names: list, title: str, output_path: str,
                          class_order: list = None) -> None:
    """
    混同行列をPNG, PDF, CSV形式で保存
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

    # PNG, PDF保存
    _plot_confusion_matrix(cm, class_names, title, str(output_path))


def _plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str, output_path: str) -> None:
    """
    混同行列を画像として保存（行ごとに正規化、0〜1の値、PNG/PDF両形式）
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
    plt.savefig(output_path + ".png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path + ".pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_cm_from_eval_results(json_path: str, n_classes: int) -> list:
    """
    評価結果JSONファイル（10fold分）を処理し、fold単位の混同行列リストを返す
    """
    data = load_eval_results(json_path)
    fold_cms = []

    for fold_data in data:
        y_true = []
        y_pred = []

        for sample in fold_data["results"]:
            pred_label = majority_vote_prediction(sample["predictions"])
            true_label = majority_vote_label(sample["labels"])
            y_pred.append(pred_label)
            y_true.append(true_label)

        cm = compute_confusion_matrix(y_true, y_pred, n_classes)
        fold_cms.append(cm)

    return fold_cms


def save_all_total_cms(param_dirs: list, sample_groups: list, output_dir,
                       class_names: list, class_order: list = None) -> None:
    """
    全パラメータのtotal混同行列を計算・保存
    """
    for param_dir in param_dirs:
        param_name = param_dir.name
        output_param_dir = output_dir / param_name

        # accumulated CMを読み込んで統合（既に並び替え済み）
        all_cms = []
        for group in sample_groups:
            csv_path = output_param_dir / group / "accumulated.csv"
            if csv_path.exists():
                cm = load_confusion_matrix(str(csv_path))
                all_cms.append(cm)

        if not all_cms:
            continue

        total_cm = accumulate_confusion_matrices(all_cms)

        output_path = output_param_dir / "total"
        save_confusion_matrix(total_cm, class_names, f"{param_name} / total", str(output_path), None)
        print(f"Total CM saved: {param_name}")
