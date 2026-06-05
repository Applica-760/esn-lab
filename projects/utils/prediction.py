import json
import os
from collections import defaultdict

import numpy as np


def _result_path(base: str) -> str:
    return base if base.endswith(".npz") else base + ".npz"


def _json_path(base: str) -> str:
    return base if base.endswith(".json") else base + ".json"


def is_valid_result_file(base: str) -> bool:
    json_path = _json_path(base)
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                return isinstance(json.load(f), list)
        except Exception:
            return False

    path = _result_path(base)
    if not os.path.exists(path):
        return False
    try:
        np.load(path, allow_pickle=False)
        return True
    except Exception:
        return False


def save_pred_results(results: list, base: str) -> None:
    path = _result_path(base)
    if d := os.path.dirname(path):
        os.makedirs(d, exist_ok=True)

    samples = [(fold["fold_index"], s) for fold in results for s in fold["results"]]
    preds = [np.asarray(s["predictions"], dtype=np.float32) for _, s in samples]
    labels = [np.asarray(s["labels"], dtype=np.float32) for _, s in samples]

    np.savez_compressed(
        path,
        fold_indices=np.array([fi for fi, _ in samples], dtype=np.int32),
        ids=np.array([s["id"] for _, s in samples]),
        lengths=np.array([p.shape[0] for p in preds], dtype=np.int32),
        predictions=np.concatenate(preds),
        labels=np.concatenate(labels),
    )


def load_pred_results(base: str) -> list:
    json_path = _json_path(base)
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        for fold in data:
            for sample in fold["results"]:
                sample["predictions"] = np.array(sample["predictions"], dtype=np.float32)
                sample["labels"] = np.array(sample["labels"], dtype=np.float32)
        return data

    data = np.load(_result_path(base), allow_pickle=False)
    splits = np.cumsum(data["lengths"])[:-1]
    preds = np.split(data["predictions"], splits)
    labels = np.split(data["labels"], splits)

    folds = defaultdict(list)
    for i, fi in enumerate(data["fold_indices"].tolist()):
        folds[fi].append({"id": str(data["ids"][i]), "predictions": preds[i], "labels": labels[i]})

    return [{"fold_index": fi, "results": ss} for fi, ss in sorted(folds.items())]
