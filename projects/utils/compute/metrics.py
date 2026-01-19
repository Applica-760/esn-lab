import re
import numpy as np


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
