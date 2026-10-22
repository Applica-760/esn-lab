from collections import defaultdict
from pathlib import Path

from projects.tasks.analysis.raw_output_viewer.judgments import summarize_votes


def validate_names(values, name, *, allow_empty=False):
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"{name} must be a list of names")
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or value in (".", "..")
            or Path(value).name != value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError(f"invalid {name}: {value!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")


def collect_group(store, judgments, group, parameter, sample_ids, identities):
    """一つのGroup・parameterだけを保持し、同じIDの真値・長さを照合する。"""
    records = defaultdict(list)
    for fold in store.list_conditions(group, parameter)["folds"]:
        samples = store.list_samples(group, parameter, fold)
        for info in sorted(samples, key=lambda r: (r["id"], r["mode"] != "train")):
            sample_id = info["id"]
            if sample_ids and sample_id not in sample_ids:
                continue
            validate_names([sample_id], "sample ID")
            identity = (info["true_label"], info["length"])
            if sample_id in identities and identities[sample_id] != identity:
                raise ValueError(f"inconsistent label or length for ID: {sample_id}")
            identities[sample_id] = identity
            mode = info["mode"]
            sample = store.get_sample(group, parameter, fold, mode, sample_id)
            judgment = judgments[mode].get_judgment(
                group, parameter, fold, mode, sample_id, info["true_label"]
            )
            votes = summarize_votes(sample["predictions"], judgment)
            values = sample["predictions"]
            records[sample_id].append(
                {
                    **info,
                    "fold_index": fold,
                    "predictions": values,
                    "judgment": judgment,
                    "votes": votes,
                    "is_correct": votes["pred_label"] == info["true_label"],
                    "out_of_range": bool((values < -1.5).any() or (values > 2.5).any()),
                }
            )
    return records


def summarize_ids(rows):
    """条件の反復回数を、parameter・ID・train/test別に集計する。"""
    totals = {}
    for row in rows:
        key = (row["parameter"], row["id"], row["mode"])
        summary = totals.setdefault(
            key,
            dict(
                parameter=row["parameter"],
                id=row["id"],
                true_label=row["true_label"],
                mode=row["mode"],
                evaluations=0,
                errors=0,
                judgment_unavailable=0,
                judgment_mismatches=0,
            ),
        )
        summary["evaluations"] += 1
        summary["errors"] += int(not row["is_correct"])
        summary["judgment_unavailable"] += int(row["judgment_status"] != "available")
        summary["judgment_mismatches"] += int(row["matches_saved_judgment"] is False)
    for summary in totals.values():
        summary["error_rate"] = summary["errors"] / summary["evaluations"]
    return sorted(
        totals.values(),
        key=lambda r: (r["parameter"], r["mode"], -r["errors"], -r["error_rate"], r["id"]),
    )
