"""
judgment_results のフィルタリングロジック（純粋関数）

CUIや他のインターフェースから独立した、再利用可能なフィルタリング処理を提供する。
"""

import random
from typing import Optional
from collections import defaultdict


# =============================================================================
# フィルタリング関数
# =============================================================================

def filter_by_true_label(judgment_results: list, label: int) -> list:
    """
    正解ラベルでフィルタ
    """
    return [r for r in judgment_results if r["true_label"] == label]


def filter_by_pred_label(judgment_results: list, label: int) -> list:
    """
    予測ラベルでフィルタ
    """
    return [r for r in judgment_results if r["pred_label"] == label]


def filter_by_correctness(judgment_results: list, is_correct: bool) -> list:
    """
    判定の正誤でフィルタ
    """
    return [r for r in judgment_results if r["is_correct"] == is_correct]


def filter_by_group(judgment_results: list, groups: list) -> list:
    """
    groupでフィルタ
    """
    return [r for r in judgment_results if r["group"] in groups]


def filter_by_fold(judgment_results: list, fold_indices: list) -> list:
    """
    fold_indexでフィルタ
    """
    return [r for r in judgment_results if r["fold_index"] in fold_indices]


def apply_filters(judgment_results: list, filter_config: dict) -> list:
    """
    複数のフィルタ条件を順次適用
    """
    results = judgment_results
    
    if filter_config.get("true_label") is not None:
        results = filter_by_true_label(results, filter_config["true_label"])
    
    if filter_config.get("pred_label") is not None:
        results = filter_by_pred_label(results, filter_config["pred_label"])
    
    if filter_config.get("is_correct") is not None:
        results = filter_by_correctness(results, filter_config["is_correct"])
    
    if filter_config.get("groups") is not None:
        results = filter_by_group(results, filter_config["groups"])
    
    if filter_config.get("fold_indices") is not None:
        results = filter_by_fold(results, filter_config["fold_indices"])
    
    return results


def sample_all(judgment_results: list) -> list:
    """
    全件を返す
    """
    return judgment_results


def sample_random(judgment_results: list, n: int, seed: Optional[int] = None) -> list:
    """
    ランダムにn件をサンプリング
    """
    if seed is not None:
        random.seed(seed)
    
    if n >= len(judgment_results):
        return judgment_results
    
    return random.sample(judgment_results, n)


def sample_first(judgment_results: list, n: int) -> list:
    """
    先頭n件を取得
    """
    return judgment_results[:n]


def apply_sampling(judgment_results: list, sampling_config: dict) -> list:
    """
    サンプリングを適用
    """
    method = sampling_config.get('method', 'all')
    n = sampling_config.get('n')
    seed = sampling_config.get('seed')
    
    if method == "all":
        return sample_all(judgment_results)
    elif method == "random":
        return sample_random(judgment_results, n, seed)
    elif method == "first":
        return sample_first(judgment_results, n)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def extract_ids(judgment_results: list) -> list:
    """
    判定結果リストからIDのリストを抽出
    """
    return list(set(r["id"] for r in judgment_results))


def extract_ids_with_metadata(judgment_results: list) -> list:
    """
    判定結果リストからID・group・fold_indexの組を抽出
    """
    return [
        {"id": r["id"], "group": r["group"], "fold_index": r["fold_index"]}
        for r in judgment_results
    ]


def group_targets_by_source(targets: list) -> dict:
    """
    targetsを(group, fold_index)でグルーピング
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for target in targets:
        grouped[target["group"]][target["fold_index"]].append(target["id"])
    
    # defaultdictを通常のdictに変換
    return {group: dict(folds) for group, folds in grouped.items()}
