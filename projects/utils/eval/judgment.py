import csv
import numpy as np
from pathlib import Path


def judge_sample_by_majority_vote(predictions, labels) -> dict:
    """
    多数決によるサンプル判定
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # フレーム単位のラベルに変換（argmax）
    pred_frames = np.argmax(predictions, axis=1)
    true_frames = np.argmax(labels, axis=1)

    # 多数決で最終ラベルを決定
    pred_label = int(np.argmax(np.bincount(pred_frames)))
    true_label = int(np.argmax(np.bincount(true_frames)))

    return {
        "pred_label": pred_label,
        "true_label": true_label,
        "is_correct": pred_label == true_label,
    }


def compute_judgment_results(pred_results: list, group: str = None) -> list:
    """
    予測結果から全サンプルの判定結果を計算
    """
    judgment_results = []

    for fold_data in pred_results:
        fold_index = fold_data["fold_index"]

        for sample in fold_data["results"]:
            judgment = judge_sample_by_majority_vote(sample["predictions"], sample["labels"])

            judgment_results.append({
                "group": group,
                "fold_index": fold_index,
                "id": sample["id"],
                "pred_label": judgment["pred_label"],
                "true_label": judgment["true_label"],
                "is_correct": judgment["is_correct"],
            })

    return judgment_results


def save_judgment_results(judgment_results: list, output_path: str) -> None:
    """
    判定結果をCSV形式で保存
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = str(output_path) + ".csv"

    fieldnames = ["group", "fold_index", "id", "pred_label", "true_label", "is_correct"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judgment_results)


def load_judgment_results(csv_path: str) -> list:
    """
    判定結果CSVを読み込み、辞書のリストとして返す
    """
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "group": row["group"],
                "fold_index": int(row["fold_index"]),
                "id": row["id"],
                "pred_label": int(row["pred_label"]),
                "true_label": int(row["true_label"]),
                "is_correct": row["is_correct"] == "True",
            })

    return results
