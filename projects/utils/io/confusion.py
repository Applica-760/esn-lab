import csv
import numpy as np
from pathlib import Path


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
    
    Args:
        cm: 混同行列
        class_names: クラス名のリスト（表示順）
        output_path: 保存先パス（拡張子なし）
        class_order: クラスの並び替え順序（Noneの場合は並び替えなし）
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
