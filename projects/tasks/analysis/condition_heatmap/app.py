import csv
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from projects.tasks.analysis.condition_heatmap.plots import render_heatmaps
from projects.tasks.analysis.raw_output_viewer.judgments import JudgmentStore, summarize_votes
from projects.utils.eval.judgment import load_judgment_results
from projects.utils.prediction import load_pred_results

LABELS = ("foraging", "rumination")
FIELDS = [
    "id",
    "true_label",
    "parameter",
    "group",
    "fold_index",
    "mode",
    "status",
    "total",
    "foraging_votes",
    "rumination_votes",
    "foraging_ratio",
    "rumination_ratio",
    "true_ratio",
    "pred_label",
    "is_correct",
    "judgment_status",
    "saved_pred_label",
    "saved_is_correct",
    "matches_saved_judgment",
    "message",
]


def _names(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list without duplicates")
    if any(
        not isinstance(v, str) or not v or v in (".", "..") or Path(v).name != v or "\\" in v
        for v in values
    ):
        raise ValueError(f"invalid {name}")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")
    return values


def _within(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path outside input directory: {path}")
    return path


def _load_npz(path):
    # 共通ローダーのzipによるメタデータの切捨てを、読込前に検出する。
    with np.load(path, allow_pickle=False) as archive:
        lengths = archive["lengths"]
        folds = archive["fold_indices"]
        ids = archive["ids"]
        if (
            lengths.ndim != 1
            or lengths.dtype.kind not in "iu"
            or (lengths <= 0).any()
            or folds.shape != lengths.shape
            or folds.dtype.kind not in "iu"
            or (folds < 0).any()
            or ids.shape != lengths.shape
            or ids.dtype.kind not in "US"
            or any(not str(v) for v in ids)
        ):
            raise ValueError("invalid NPZ sample metadata")
        shape = (int(lengths.sum()), 2)
        for name in ("predictions", "labels"):
            values = archive[name]
            if values.shape != shape or values.dtype.kind not in "fiu":
                raise ValueError(f"invalid {name} shape or dtype")
    return load_pred_results(str(path))


def collect_rows(cfg):
    root = Path(cfg.pred_result_dir).resolve()
    saved_root = Path(cfg.judgment_result_dir).resolve()
    groups = _names(cfg.groups, "groups")
    parameters = _names(cfg.parameters, "parameters")
    modes = _names(cfg.modes, "modes")
    if set(modes) - {"train", "test"}:
        raise ValueError("modes must be train or test")
    folds = cfg.fold_indices
    if (
        not isinstance(folds, list)
        or not folds
        or any(type(f) is not int or f < 0 for f in folds)
        or len(set(folds)) != len(folds)
    ):
        raise ValueError("fold_indices must be nonnegative unique integers")
    stores = {(p, m): JudgmentStore(saved_root) for p in parameters for m in modes}
    samples = {}
    identities = {}
    file_states = {}

    for parameter in parameters:
        for group in groups:
            for mode in modes:
                path = _within(root, group, parameter, f"{mode}_results.npz")
                key = (parameter, group, mode)
                if not path.is_file():
                    file_states[key] = ("missing_file", "prediction NPZ not found")
                    continue
                try:
                    results = _load_npz(path)
                except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
                    file_states[key] = ("invalid_file", str(exc))
                    continue
                file_states[key] = ("missing_sample", "ID not present in this condition")
                for fold in results:
                    if fold["fold_index"] not in folds:
                        continue
                    for sample in fold["results"]:
                        sample_id = sample["id"]
                        cell = (parameter, group, fold["fold_index"], mode, sample_id)
                        if cell in samples:
                            raise ValueError(f"duplicate ID in condition: {cell}")
                        predictions = np.asarray(sample["predictions"])
                        labels = np.asarray(sample["labels"])
                        row = dict(
                            id=sample_id,
                            parameter=parameter,
                            group=group,
                            fold_index=fold["fold_index"],
                            mode=mode,
                        )
                        samples[cell] = row
                        if (
                            not np.isfinite(predictions).all()
                            or not np.isfinite(labels).all()
                            or not np.isin(labels, [0, 1]).all()
                            or not (labels.sum(axis=1) == 1).all()
                            or not (labels == labels[0]).all()
                        ):
                            row.update(status="invalid_data", message="invalid output or labels")
                            continue
                        true_label = LABELS[int(labels[0].argmax())]
                        identity = (true_label, len(predictions))
                        if sample_id in identities and identities[sample_id] != identity:
                            raise ValueError(f"inconsistent label or length for ID: {sample_id}")
                        identities[sample_id] = identity
                        judgment = stores[parameter, mode].get_judgment(
                            group, parameter, fold["fold_index"], mode, sample_id, true_label
                        )
                        votes = summarize_votes(predictions, judgment)
                        row.update(
                            true_label=true_label,
                            status="available",
                            total=votes["total"],
                            foraging_votes=votes["counts"]["foraging"],
                            rumination_votes=votes["counts"]["rumination"],
                            foraging_ratio=votes["ratios"]["foraging"],
                            rumination_ratio=votes["ratios"]["rumination"],
                            true_ratio=votes["ratios"][true_label],
                            pred_label=votes["pred_label"],
                            is_correct=votes["pred_label"] == true_label,
                            matches_saved_judgment=votes["matches_saved_judgment"],
                        )
                        if judgment["status"] == "invalid_data":
                            row["status"] = "invalid_judgment"
                        elif votes["matches_saved_judgment"] is False:
                            row["status"] = "judgment_mismatch"
                        _add_judgment(row, judgment)
                # 全条件の生出力を保持せず、集計済みの行だけ残す。
                del results

    # 予測が欠けていて保存判定だけ存在するIDも一覧に残す。
    for parameter in parameters:
        for mode in modes:
            path = _within(saved_root, parameter, f"judgment_results_{mode}.csv")
            try:
                saved_rows = load_judgment_results(path)
            except (OSError, ValueError, KeyError, TypeError, csv.Error):
                continue  # 各セルにはJudgmentStoreの欠落・不正状態を記録する。
            for saved in saved_rows:
                if saved["group"] not in groups or saved["fold_index"] not in folds:
                    continue
                sample_id = saved["id"]
                if not sample_id or saved["true_label"] not in (0, 1):
                    continue
                label = LABELS[saved["true_label"]]
                judgment = stores[parameter, mode].get_judgment(
                    saved["group"], parameter, saved["fold_index"], mode, sample_id, label
                )
                if judgment["status"] == "available":
                    identities.setdefault(sample_id, (label, None))

    ids = set(identities) | {key[-1] for key in samples}
    if not ids:
        raise ValueError("no usable IDs found in predictions or saved judgments")
    rows = []
    for sample_id in sorted(ids, key=lambda i: (identities.get(i, ("", None))[0], i)):
        true_label = identities.get(sample_id, ("", None))[0]
        for parameter in parameters:
            for group in groups:
                for fold in sorted(folds):
                    for mode in modes:
                        cell = (parameter, group, fold, mode, sample_id)
                        row = samples.get(cell)
                        if row is None:
                            status, message = file_states[(parameter, group, mode)]
                            row = dict(
                                id=sample_id,
                                true_label=true_label,
                                parameter=parameter,
                                group=group,
                                fold_index=fold,
                                mode=mode,
                                status=status,
                                message=message,
                            )
                        row.setdefault("true_label", true_label)
                        if "judgment_status" not in row:
                            judgment = stores[parameter, mode].get_judgment(
                                group, parameter, fold, mode, sample_id, true_label
                            )
                            _add_judgment(row, judgment)
                        rows.append(row)
    return rows


def _add_judgment(row, judgment):
    row["judgment_status"] = judgment["status"]
    if judgment["status"] == "available":
        row["saved_pred_label"] = judgment["pred_label"]
        row["saved_is_correct"] = judgment["is_correct"]
    elif "message" in judgment:
        row["message"] = "; ".join(filter(None, [row.get("message"), judgment["message"]]))


def main(cfg):
    output = Path(cfg.output_dir).resolve()
    for value in (cfg.pred_result_dir, cfg.judgment_result_dir):
        source = Path(value).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output directory must not overlap an input directory")
    rows = collect_rows(cfg)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "condition_cells.csv"
    if path.is_symlink():
        raise ValueError("output CSV must not be a symbolic link")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} cells: {path}")
    paths = render_heatmaps(rows, output, cfg.groups, cfg.fold_indices, cfg.parameters, cfg.modes)
    print(f"{len(paths)} heatmaps: {output}")
