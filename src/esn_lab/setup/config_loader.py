# esn_lab/setup/config_loader.py
import yaml
from pathlib import Path
from typing import Any

from esn_lab.setup.config import DataSourceCfg


def _canonical_mode(mode: str) -> str:
    """Map CLI mode to canonical config branch/dir names.

    - pred -> predict
    - eval -> evaluate
    - otherwise passthrough
    """
    return {"pred": "predict", "eval": "evaluate"}.get(mode, mode)

def load_and_merge_configs(mode: str, variant: str) -> dict:
    base_yaml_path = Path("configs/base.yaml")
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {base_yaml_path}")

    canon = _canonical_mode(mode)
    yaml_path = Path("configs") / canon / f"{variant}.yaml"
    with open(yaml_path, "r") as f:
        mode_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {yaml_path}")

    base_cfg.setdefault(canon, {})[variant] = mode_cfg

    return base_cfg


# ========================================================================
# 設定アクセス・正規化ヘルパー関数
# ========================================================================

def safe_get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """オブジェクトの属性または辞書キーを安全に取得する。
    
    Args:
        obj: 取得対象のオブジェクトまたは辞書
        key: 属性名またはキー
        default: デフォルト値
    
    Returns:
        取得した値、存在しない場合はdefault
    """
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except Exception:
        try:
            return obj.get(key, default)  # type: ignore[attr-defined]
        except Exception:
            return default


def normalize_data_source(data_source: Any) -> DataSourceCfg | None:
    """data_source設定を正規化する。
    
    辞書形式の場合はDataSourceCfgオブジェクトに変換する。
    
    Args:
        data_source: DataSourceCfgオブジェクトまたは辞書、またはNone
    
    Returns:
        正規化されたDataSourceCfgオブジェクト、またはNone
    """
    if data_source is None:
        return None
    
    if isinstance(data_source, dict):
        return DataSourceCfg(
            type=data_source.get("type", "csv"),
            csv_dir=data_source.get("csv_dir"),
            npy_dir=data_source.get("npy_dir")
        )
    
    # 既にDataSourceCfgオブジェクトの場合はそのまま返す
    return data_source


def create_eval_tenfold_config(
    base_tenfold_cfg: Any,
    grid_eval_tenfold_cfg: Any,
    csv_dir_str: str,
    data_source_eval: Any,
    experiment_name_eval: str,
    eval_workers: int,
    eval_parallel: bool,
) -> Any:
    """評価用のtenfold設定を作成する。
    
    grid_eval_tenfold_cfgが存在する場合はそれをベースに不足項目を補完し、
    存在しない場合は新規に作成する。元のcfgを変更せず、新しい設定を返す。
    
    Args:
        base_tenfold_cfg: ベースとなるtenfold設定
        grid_eval_tenfold_cfg: grid設定内の評価設定（あればそれを優先）
        csv_dir_str: CSVディレクトリパス
        data_source_eval: データソース設定
        experiment_name_eval: 実験名
        eval_workers: ワーカー数
        eval_parallel: 並列実行フラグ
    
    Returns:
        EvaluateTenfoldCfg: 評価用tenfold設定
    """
    from esn_lab.setup.config import EvaluateTenfoldCfg
    
    if grid_eval_tenfold_cfg:
        # ユーザ指定の設定をベースに、不足項目を補完
        return EvaluateTenfoldCfg(
            csv_dir=safe_get_attr(grid_eval_tenfold_cfg, "csv_dir", None) or csv_dir_str,
            data_source=safe_get_attr(grid_eval_tenfold_cfg, "data_source", None) or data_source_eval,
            experiment_name=safe_get_attr(grid_eval_tenfold_cfg, "experiment_name", None) or experiment_name_eval,
            workers=safe_get_attr(grid_eval_tenfold_cfg, "workers", None) or eval_workers,
            parallel=safe_get_attr(grid_eval_tenfold_cfg, "parallel", None) if safe_get_attr(grid_eval_tenfold_cfg, "parallel", None) is not None else eval_parallel,
            search_space=safe_get_attr(grid_eval_tenfold_cfg, "search_space", None),
        )
    else:
        # 新規作成
        return EvaluateTenfoldCfg(
            csv_dir=csv_dir_str,
            data_source=data_source_eval,
            experiment_name=experiment_name_eval,
            workers=eval_workers,
            parallel=eval_parallel,
        )


def create_eval_summary_config(
    grid_eval_summary_cfg: Any,
    experiment_name_eval: str,
    search_space: dict | None,
) -> Any:
    """評価用のsummary設定を作成する。
    
    grid_eval_summary_cfgが存在する場合はそれをベースに不足項目を補完し、
    存在しない場合は新規に作成する。元のcfgを変更せず、新しい設定を返す。
    
    Args:
        grid_eval_summary_cfg: grid設定内のsummary設定（あればそれを優先）
        experiment_name_eval: 実験名
        search_space: パラメータ探索空間
    
    Returns:
        EvaluateSummaryCfg: 評価用summary設定
    """
    from esn_lab.setup.config import EvaluateSummaryCfg
    
    if grid_eval_summary_cfg:
        # ユーザ指定の設定をベースに、不足項目を補完
        exp_name = safe_get_attr(grid_eval_summary_cfg, "experiment_name", None) or experiment_name_eval
        vary_param = safe_get_attr(grid_eval_summary_cfg, "vary_param", None) or "Nx"
        vary_values = safe_get_attr(grid_eval_summary_cfg, "vary_values", None)
        
        # vary_valuesが未指定でsearch_spaceがある場合、自動補完を試みる
        if vary_values in (None, []) and search_space:
            candidates = []
            if isinstance(search_space, dict):
                if f"model.{vary_param}" in search_space:
                    candidates = list(search_space[f"model.{vary_param}"])
                elif vary_param in search_space:
                    candidates = list(search_space[vary_param])
            if candidates:
                vary_values = candidates
        
        return EvaluateSummaryCfg(
            experiment_name=exp_name,
            results_csv=safe_get_attr(grid_eval_summary_cfg, "results_csv", None),
            predictions_csv=safe_get_attr(grid_eval_summary_cfg, "predictions_csv", None),
            metric=safe_get_attr(grid_eval_summary_cfg, "metric", None) or "majority_acc",
            vary_param=vary_param,
            vary_values=vary_values,
            filters=safe_get_attr(grid_eval_summary_cfg, "filters", None),
            agg=safe_get_attr(grid_eval_summary_cfg, "agg", None) or "mean",
            output_dir=safe_get_attr(grid_eval_summary_cfg, "output_dir", None),
            fmt=safe_get_attr(grid_eval_summary_cfg, "fmt", None) or ".3f",
            title=safe_get_attr(grid_eval_summary_cfg, "title", None),
            dpi=safe_get_attr(grid_eval_summary_cfg, "dpi", None) or 150,
        )
    else:
        # 新規作成: search_spaceから推測
        vary_param = "Nx"
        if search_space and isinstance(search_space, dict) and len(search_space) >= 1:
            fields = []
            for k in search_space.keys():
                if isinstance(k, str) and k.startswith("model."):
                    fields.append(k.split(".", 1)[1])
            if len(fields) == 1:
                vary_param = fields[0]
            elif "Nx" in fields:
                vary_param = "Nx"
            elif len(fields) > 1:
                vary_param = fields[0]
        
        return EvaluateSummaryCfg(
            experiment_name=experiment_name_eval,
            vary_param=vary_param,
        )