import numpy as np


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

    Args:
        judgment_results: load_judgment_results()で読み込んだデータ
            [{"group": "a", "fold_index": 0, "id": ..., "pred_label": ..., "true_label": ..., "is_correct": ...}, ...]
        n_classes: クラス数
        fold_index: 特定foldのみ計算する場合に指定（Noneなら全て）
        group: 特定groupのみ計算する場合に指定（Noneなら全て）

    Returns:
        混同行列 [n_classes, n_classes]
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


def save_all_total_cms(param_dirs: list, sample_groups: list, output_dir,
                       class_names: list, class_order: list = None) -> None:
    """
    全パラメータのtotal混同行列を計算・保存
    """
    from projects.utils.io.confusion import load_confusion_matrix_csv, save_confusion_matrix_csv
    from projects.utils.plot.plot_confusion import plot_confusion_matrix
    
    for param_dir in param_dirs:
        param_name = param_dir.name
        output_param_dir = output_dir / param_name

        # accumulated CMを読み込んで統合（既に並び替え済み）
        all_cms = []
        for group in sample_groups:
            csv_path = output_param_dir / group / "accumulated.csv"
            if csv_path.exists():
                cm = load_confusion_matrix_csv(str(csv_path))
                all_cms.append(cm)

        if not all_cms:
            continue

        total_cm = np.sum(all_cms, axis=0)

        output_path = output_param_dir / "total"
        # CSV保存
        save_confusion_matrix_csv(total_cm, class_names, str(output_path), class_order)
        # プロット保存
        plot_confusion_matrix(total_cm, class_names, f"{param_name} / total", str(output_path), class_order)
        print(f"Total CM saved: {param_name}")
