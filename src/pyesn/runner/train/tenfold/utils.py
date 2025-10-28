from itertools import product
from typing import List, Tuple

def flatten_search_space(ss: dict[str, list] | None) -> list[tuple[dict, str]]:
    """
    ハイパーパラメータの検索空間(辞書)を、
    (パラメータ辞書, パラメータタグ文字列)のリストに変換する。

    仕様:
    - 入力キーは 'model.Nx' 形式を推奨するが、'Nx' のような素のキーも許容する。
    - タグはキー名の昇順で安定化して生成する（再現性のため）。
    """
    if not ss:
        return [({}, "default")]

    # 厳密化: キーは 'model.' で始まることを要求し、フィールド名へ正規化
    norm_items: list[tuple[str, List]] = []
    for raw_k, vals in ss.items():
        k = str(raw_k)
        if not k.startswith("model."):
            raise ValueError(f"search_space key must start with 'model.': {k}")
        field = k.split(".", 1)[1]
        norm_items.append((field, list(vals)))

    # 安定順序（キー名昇順）で積を取る
    norm_items.sort(key=lambda x: x[0])
    keys = [k for k, _ in norm_items]
    lists = [v for _, v in norm_items]

    combos: list[Tuple[dict, str]] = []
    for values in product(*lists):
        d = {k: v for k, v in zip(keys, values)}
        tag = "_".join([f"{k}={v}" for k, v in d.items()])
        combos.append((d, tag))

    return combos