import numpy as np


def judge_sample_by_majority_vote(predictions, labels) -> dict:
    """
    多数決によるサンプル判定

    Args:
        predictions: モデルの予測出力 [時刻, クラス数]
        labels: 正解ラベル [時刻, クラス数] (one-hot形式)

    Returns:
        {
            "pred_label": int,   # 予測ラベル（多数決）
            "true_label": int,   # 正解ラベル（多数決）
            "is_correct": bool,  # 一致判定
        }
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


def compute_judgment_results(eval_results: list, group: str = None) -> list:
    """
    評価結果から全サンプルの判定結果を計算

    Args:
        eval_results: load_eval_results()で読み込んだデータ
            [
                {"fold_index": 0, "results": [{"id": ..., "predictions": ..., "labels": ...}, ...]},
                ...
            ]
        group: サンプル群の識別子（a, b, c, ...）

    Returns:
        [
            {"group": "a", "fold_index": 0, "id": "08_013319", "pred_label": 1, "true_label": 1, "is_correct": True},
            ...
        ]
    """
    judgment_results = []

    for fold_data in eval_results:
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
