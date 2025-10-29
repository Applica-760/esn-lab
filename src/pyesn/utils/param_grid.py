from itertools import product
from typing import List, Tuple


def flatten_search_space(ss: dict[str, list] | None) -> list[tuple[dict, str]]:
    """
    共通のハイパーパラメータ展開ユーティリティ。

    入力:
      - ss: {"model.<field>": [values, ...], ...}

    出力:
      - [(overrides: dict, tag: str), ...]

    仕様:
      - キーは 'model.' で始まることを要求し、モデルフィールド名へ正規化
      - タグはキー名昇順で安定化
    """
    if not ss:
        return [({}, "default")]

    norm_items: list[tuple[str, List]] = []
    for raw_k, vals in ss.items():
        k = str(raw_k)
        if not k.startswith("model."):
            raise ValueError(f"search_space key must start with 'model.': {k}")
        field = k.split(".", 1)[1]
        norm_items.append((field, list(vals)))

    norm_items.sort(key=lambda x: x[0])
    keys = [k for k, _ in norm_items]
    lists = [v for _, v in norm_items]

    combos: list[Tuple[dict, str]] = []
    for values in product(*lists):
        d = {k: v for k, v in zip(keys, values)}
        tag = "_".join([f"{k}={v}" for k, v in d.items()])
        combos.append((d, tag))

    return combos
