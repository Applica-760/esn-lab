from collections import defaultdict
from pathlib import Path

import numpy as np

from projects.utils.prediction import is_valid_result_file, load_pred_results

CLASS_NAMES = {0: "other", 1: "foraging", 2: "rumination"}
CLASS_ORDER = ["foraging", "rumination", "other"]
PREDICTED_CLASS_NAMES = ["foraging", "rumination"]


def summarize_sample(
    predictions, labels, warmup_ratio: float
) -> tuple[str, float, int, np.ndarray]:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(f"expected predictions shape (timesteps, 2), got {predictions.shape}")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(f"expected labels shape (timesteps, 3), got {labels.shape}")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not 0 <= warmup_ratio < 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

    warmup = int(len(predictions) * warmup_ratio)
    predictions = predictions[warmup:]
    labels = labels[warmup:]
    if len(predictions) == 0:
        raise ValueError("no frames remain after warmup")

    frame_classes = np.argmax(labels, axis=1)
    if not np.all(frame_classes == frame_classes[0]):
        raise ValueError("labels must be single-class over all timesteps")
    true_label = CLASS_NAMES[int(frame_classes[0])]
    margin = float(np.abs(predictions[:, 0] - predictions[:, 1]).mean())
    return true_label, margin, len(predictions), predictions


def collect_samples(
    pred_result_dir: Path,
    groups: list[str],
    warmup_ratio: float,
    fold_indices: list[int] | None = None,
) -> dict[str, list[dict]]:
    samples_by_param = defaultdict(list)
    selected_fold_indices = set(fold_indices or [])
    for group in groups:
        group_dir = pred_result_dir / group
        if not group_dir.exists():
            print(f"skip (group not found): {group_dir}")
            continue

        for param_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            result_base = param_dir / "test_results"
            if not is_valid_result_file(str(result_base)):
                print(f"skip (result not found): {result_base}")
                continue

            for fold_data in load_pred_results(str(result_base)):
                fold_index = fold_data["fold_index"]
                if selected_fold_indices and fold_index not in selected_fold_indices:
                    continue
                for sample in fold_data["results"]:
                    true_label, margin, frames, predictions = summarize_sample(
                        sample["predictions"], sample["labels"], warmup_ratio
                    )
                    samples_by_param[param_dir.name].append(
                        {
                            "group": group,
                            "fold_index": fold_index,
                            "id": sample["id"],
                            "true_label": true_label,
                            "margin": margin,
                            "frames_after_warmup": frames,
                            "predictions": predictions,
                        }
                    )
    return samples_by_param


def margin_rows(samples: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in sample.items() if key != "predictions"}
        for sample in samples
    ]
