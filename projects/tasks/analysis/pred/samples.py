import numpy as np


def strip_warmup(predictions, labels, warmup_ratio):
    warmup = int(len(predictions) * warmup_ratio)
    return predictions[warmup:], labels[warmup:]


def normalize_fold_samples(fold_data, warmup_ratio):
    """fold内のサンプルをwarmup除外し、真値・予測クラスを付与する。"""
    samples = []
    for sample in fold_data["results"]:
        predictions, labels = strip_warmup(sample["predictions"], sample["labels"], warmup_ratio)
        true_frames = np.argmax(labels, axis=1)
        true_label = int(np.argmax(np.bincount(true_frames)))
        pred_frames = np.argmax(predictions, axis=1)
        pred_label = int(np.argmax(np.bincount(pred_frames, minlength=predictions.shape[1])))
        samples.append(
            {
                "predictions": predictions,
                "labels": labels,
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )
    return samples
