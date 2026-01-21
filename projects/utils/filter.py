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
    
    Args:
        judgment_results: 判定結果リスト
        label: フィルタする正解ラベル（0: other, 1: foraging, 2: rumination）
    
    Returns:
        フィルタ後の判定結果リスト
    """
    return [r for r in judgment_results if r["true_label"] == label]


def filter_by_pred_label(judgment_results: list, label: int) -> list:
    """
    予測ラベルでフィルタ
    
    Args:
        judgment_results: 判定結果リスト
        label: フィルタする予測ラベル
    
    Returns:
        フィルタ後の判定結果リスト
    """
    return [r for r in judgment_results if r["pred_label"] == label]


def filter_by_correctness(judgment_results: list, is_correct: bool) -> list:
    """
    判定の正誤でフィルタ
    
    Args:
        judgment_results: 判定結果リスト
        is_correct: True=正解のみ, False=不正解のみ
    
    Returns:
        フィルタ後の判定結果リスト
    """
    return [r for r in judgment_results if r["is_correct"] == is_correct]


def filter_by_group(judgment_results: list, groups: list) -> list:
    """
    groupでフィルタ
    
    Args:
        judgment_results: 判定結果リスト
        groups: フィルタするgroupのリスト（例: ["a", "b", "c"]）
    
    Returns:
        フィルタ後の判定結果リスト
    """
    return [r for r in judgment_results if r["group"] in groups]


def filter_by_fold(judgment_results: list, fold_indices: list) -> list:
    """
    fold_indexでフィルタ
    
    Args:
        judgment_results: 判定結果リスト
        fold_indices: フィルタするfold_indexのリスト（例: [0, 1, 2]）
    
    Returns:
        フィルタ後の判定結果リスト
    """
    return [r for r in judgment_results if r["fold_index"] in fold_indices]


def apply_filters(judgment_results: list, filter_config: dict) -> list:
    """
    複数のフィルタ条件を順次適用
    
    Args:
        judgment_results: 判定結果リスト
        filter_config: フィルタ条件の辞書
            {
                "true_label": Optional[int],      # 正解ラベル
                "pred_label": Optional[int],      # 予測ラベル
                "is_correct": Optional[bool],     # 正誤
                "groups": Optional[list[str]],    # group
                "fold_indices": Optional[list[int]],  # fold
            }
    
    Returns:
        全フィルタ適用後の判定結果リスト
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


# =============================================================================
# サンプリング関数
# =============================================================================

def sample_all(judgment_results: list) -> list:
    """
    全件を返す（サンプリングなし）
    """
    return judgment_results


def sample_random(judgment_results: list, n: int, seed: Optional[int] = None) -> list:
    """
    ランダムにn件をサンプリング
    
    Args:
        judgment_results: 判定結果リスト
        n: サンプリング件数
        seed: 乱数シード（再現性のため）
    
    Returns:
        サンプリング後の判定結果リスト
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


def apply_sampling(judgment_results: list, method: str, n: Optional[int] = None, 
                   seed: Optional[int] = None) -> list:
    """
    サンプリングを適用
    
    Args:
        judgment_results: 判定結果リスト
        method: "all", "random", "first" のいずれか
        n: サンプリング件数（method="all"の場合は無視）
        seed: 乱数シード
    
    Returns:
        サンプリング後の判定結果リスト
    """
    if method == "all":
        return sample_all(judgment_results)
    elif method == "random":
        return sample_random(judgment_results, n, seed)
    elif method == "first":
        return sample_first(judgment_results, n)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# =============================================================================
# ID抽出
# =============================================================================

def extract_ids(judgment_results: list) -> list:
    """
    判定結果リストからIDのリストを抽出
    
    Returns:
        IDのリスト（重複なし）
    """
    return list(set(r["id"] for r in judgment_results))


def extract_ids_with_metadata(judgment_results: list) -> list:
    """
    判定結果リストからID・group・fold_indexの組を抽出
    
    プロット時にeval_results.jsonから該当データを取得するために使用。
    
    Returns:
        [{"id": str, "group": str, "fold_index": int}, ...]
    """
    return [
        {"id": r["id"], "group": r["group"], "fold_index": r["fold_index"]}
        for r in judgment_results
    ]


# =============================================================================
# グルーピング（プロット実行最適化用）
# =============================================================================

def group_targets_by_source(targets: list) -> dict:
    """
    targetsを(group, fold_index)でグルーピング
    
    同じeval_results.jsonから取得するデータをまとめることで、
    JSONロード回数を最小化する。
    
    Args:
        targets: [{"id": str, "group": str, "fold_index": int}, ...]
    
    Returns:
        {group: {fold_index: [id, id, ...], ...}, ...}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for target in targets:
        grouped[target["group"]][target["fold_index"]].append(target["id"])
    
    # defaultdictを通常のdictに変換
    return {group: dict(folds) for group, folds in grouped.items()}
