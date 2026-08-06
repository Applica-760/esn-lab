from collections import defaultdict
from pathlib import Path

from projects.utils.eval.filter import apply_filters
from projects.utils.eval.judgment import load_judgment_results
from projects.utils.prediction import is_valid_result_file, load_pred_results


def _key_context(param_name: str, mode: str, group: str, fold_index: int, sample_id: str) -> str:
    return (
        f"param_name={param_name!r}, mode={mode!r}, group={group!r}, "
        f"fold_index={fold_index}, id={sample_id!r}"
    )


def load_filtered_prediction_samples(
    param_name: str,
    mode: str,
    judge_dir: Path,
    pred_result_dir: Path,
    filters: dict,
) -> list:
    """判定結果をfilterし、予測サンプルと複合キーで結合する。"""
    judgment_csv_path = Path(judge_dir) / param_name / f"judgment_results_{mode}.csv"
    if not judgment_csv_path.exists():
        return []

    judgment_results = load_judgment_results(judgment_csv_path)
    filtered_results = apply_filters(judgment_results, filters)
    if not filtered_results:
        return []

    grouped_results = defaultdict(list)
    judgment_keys = set()
    for result in filtered_results:
        key = (result["group"], result["fold_index"], result["id"])
        if key in judgment_keys:
            context = _key_context(param_name, mode, *key)
            raise ValueError(f"duplicate judgment key: {context}")
        judgment_keys.add(key)
        grouped_results[result["group"]].append(result)

    joined_samples = []
    for group, results in grouped_results.items():
        result_base = str(Path(pred_result_dir) / group / param_name / f"{mode}_results")
        if not is_valid_result_file(result_base):
            continue

        predictions_by_key = {}
        for fold_data in load_pred_results(result_base):
            fold_index = fold_data["fold_index"]
            for sample in fold_data["results"]:
                key = (fold_index, sample["id"])
                if key in predictions_by_key:
                    context = _key_context(param_name, mode, group, *key)
                    raise ValueError(f"duplicate prediction key: {context}")
                predictions_by_key[key] = sample

        for result in results:
            key = (result["fold_index"], result["id"])
            sample = predictions_by_key.get(key)
            if sample is None:
                context = _key_context(param_name, mode, group, *key)
                raise ValueError(f"missing prediction key: {context}")

            joined_samples.append(
                {
                    **result,
                    "predictions": sample["predictions"],
                    "labels": sample["labels"],
                }
            )

    return joined_samples
