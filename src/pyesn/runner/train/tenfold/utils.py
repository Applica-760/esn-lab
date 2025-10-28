from itertools import product
from pyesn.pipeline.tenfold_util import load_10fold_csv_mapping, read_data_from_csvs

def flatten_search_space(ss: dict[str, list] | None) -> list[tuple[dict, str]]:
    """
    ハイパーパラメータの検索空間(辞書)を、
    (パラメータ辞書, パラメータタグ文字列)のリストに変換する。
    """
    if not ss:
        return [({}, "default")]

    items = []
    for k, vals in ss.items():
        if not k.startswith("model."):
            raise ValueError(f"search_space key must start with 'model.': {k}")
        field = k.split(".", 1)[1]
        items.append((field, list(vals)))

    keys = [k for k, _ in items]
    lists = [v for _, v in items]
    
    combos = []
    for values in product(*lists):
        d = {k: v for k, v in zip(keys, values)}
        tag = "_".join([f"{k}={v}" for k, v in d.items()])
        combos.append((d, tag))
        
    return combos