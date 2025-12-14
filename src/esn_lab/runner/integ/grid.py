from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path
from copy import deepcopy

from esn_lab.runner.train.tenfold.main import run_tenfold
from esn_lab.utils.param_grid import flatten_search_space
from esn_lab.runner.eval.evaluate import tenfold_evaluate, summary_evaluate
from esn_lab.setup.config import (
    Evaluate,
    TrainTenfoldCfg,
)
from esn_lab.setup.config_loader import (
    safe_get_attr,
    normalize_data_source,
    create_eval_tenfold_config,
    create_eval_summary_config,
)


def run_grid(cfg) -> None:
    """ハイパーパラメタグリッドを総当たりし、各組み合わせでシンプルな tenfold 学習を実行する。

        期待する設定（cfg.integ.grid.train）:
            - data_source: データソース設定（type: "csv" または "npy"、新方式・推奨）
            - csv_dir: 10-fold分割済みCSVのディレクトリ（後方互換、data_sourceがない場合）
            - experiment_name: 実験名（必須）
            - workers: 並列ワーカ数（1で逐次、2以上で並列）
            - search_space: {"model.<field>": [values, ...], ...}
    """
    integ = getattr(cfg, "integ", None)
    if integ is None or getattr(integ, "grid", None) is None:
        raise ValueError("Config 'integ.grid' not found.")

    grid_cfg = getattr(integ, "grid")

    # 新形式: integ.grid が単一の grid.yaml 形式を持つ
    # structure: base, param_grid (or search_space), train, eval, summary
    is_new_grid_schema = (safe_get_attr(grid_cfg, "param_grid") is not None) or (safe_get_attr(grid_cfg, "base") is not None)

    if is_new_grid_schema:
        base = safe_get_attr(grid_cfg, "base") or {}
        param_grid = safe_get_attr(grid_cfg, "param_grid") or safe_get_attr(grid_cfg, "search_space")
        train_template = safe_get_attr(grid_cfg, "train") or {}
        grid_eval_cfg = safe_get_attr(grid_cfg, "eval") or {}
        summary_template = safe_get_attr(grid_cfg, "summary") or {}

        # 組み合わせ展開
        ss = param_grid
        combos: List[Tuple[Dict, str]] = flatten_search_space(ss)

        # experiment_name は必須
        experiment_name = safe_get_attr(grid_cfg, "experiment_name") or safe_get_attr(train_template, "experiment_name") or safe_get_attr(base, "experiment_name")
        if not experiment_name:
            raise ValueError("integ.grid requires 'experiment_name'.")
        
        print(f"[INFO] Using experiment: {experiment_name}")

        # tenfold 設定を base + train_template から作る
        csv_dir = safe_get_attr(train_template, "csv_dir") or safe_get_attr(base, "csv_dir")
        data_source = safe_get_attr(train_template, "data_source") or safe_get_attr(base, "data_source")
        
        # data_source を正規化
        data_source = normalize_data_source(data_source)
        
        workers = safe_get_attr(train_template, "workers") or safe_get_attr(base, "workers") or 1
        skip_existing = safe_get_attr(train_template, "skip_existing")
        if skip_existing is None:
            skip_existing = safe_get_attr(base, "skip_existing")
        if skip_existing is None:
            skip_existing = True
        force_retrain = safe_get_attr(train_template, "force_retrain") or safe_get_attr(base, "force_retrain") or False

        train_cfg = TrainTenfoldCfg(
            csv_dir=csv_dir,
            data_source=data_source,
            experiment_name=experiment_name,
            workers=workers,
            skip_existing=skip_existing,
            force_retrain=force_retrain,
            search_space=None
        )

    else:
        # 既存単純形: integ.grid.train をそのまま利用
        train_cfg = getattr(grid_cfg, "train", None)
        grid_eval_cfg = getattr(grid_cfg, "eval", None)
        if train_cfg is None:
            raise ValueError("Config 'integ.grid.train' is required.")

        # search_space をフラット化
        ss = getattr(train_cfg, "search_space", None)
        combos: List[Tuple[Dict, str]] = flatten_search_space(ss)

    # 各ハイパラセットごとに単独の tenfold 学習を実行
    auto_workers = int(getattr(train_cfg, "workers", 1) or 1)
    parallel = auto_workers > 1
    max_workers = auto_workers

    # 評価に必要な共通パスは、integ.grid.eval があればそれを優先
    # なければ学習設定から補完
    if grid_eval_cfg and getattr(grid_eval_cfg, "tenfold", None):
        csv_dir_str: str = getattr(grid_eval_cfg.tenfold, "csv_dir", None) or getattr(train_cfg, "csv_dir")
        data_source_eval = getattr(grid_eval_cfg.tenfold, "data_source", None) or getattr(train_cfg, "data_source", None)
        experiment_name_eval: str | None = getattr(grid_eval_cfg.tenfold, "experiment_name", None) or getattr(train_cfg, "experiment_name", None)
        if not experiment_name_eval:
            raise ValueError("integ.grid.eval requires 'experiment_name'.")
        eval_workers: int = int(getattr(grid_eval_cfg.tenfold, "workers", None) or getattr(train_cfg, "workers", 1) or 1)
        eval_parallel: bool = bool(getattr(grid_eval_cfg.tenfold, "parallel", True))
    else:
        csv_dir_str = getattr(train_cfg, "csv_dir")
        data_source_eval = getattr(train_cfg, "data_source", None)
        experiment_name_eval = getattr(train_cfg, "experiment_name", None)
        eval_workers = auto_workers
        eval_parallel = True
    
    # data_source を正規化
    data_source_eval = normalize_data_source(data_source_eval)
    
    if not experiment_name_eval:
        raise ValueError("integ.grid requires 'experiment_name'.")

    for overrides, tag in combos:
        print("=" * 50)
        print(f"[GRID] param set: {tag}")
        print("=" * 50)
        run_tenfold(
            cfg,
            overrides=overrides,
            tenfold_cfg=train_cfg,
            parallel=parallel,
            max_workers=max_workers,
        )

    # 学習直後に、対応パラメタ集合のみ tenfold 評価を実行する（ディレクトリ全走査を回避）
        # 元のcfgを変更せず、評価用の新しいcfgを作成
        eval_cfg = deepcopy(cfg)
        
        if getattr(eval_cfg, "evaluate", None) is None:
            eval_cfg.evaluate = Evaluate(run=None, tenfold=None, summary=None)
        
        # tenfold 評価の設定を作成（元のcfgは変更しない）
        grid_eval_tenfold = getattr(grid_eval_cfg, "tenfold", None) if grid_eval_cfg else None
        eval_cfg.evaluate.tenfold = create_eval_tenfold_config(
            base_tenfold_cfg=train_cfg,
            grid_eval_tenfold_cfg=grid_eval_tenfold,
            csv_dir_str=csv_dir_str,
            data_source_eval=data_source_eval,
            experiment_name_eval=experiment_name_eval,
            eval_workers=eval_workers,
            eval_parallel=eval_parallel,
        )
        
        # このセットだけを評価対象にするため search_space を1要素で付与
        # overrides は {"Nx":..,"density":..} 形式。search_space は "model." 接頭を要求。
        one_search = {f"model.{k}": [v] for k, v in (overrides or {}).items()}
        eval_cfg.evaluate.tenfold.search_space = one_search if one_search else None
        
        print("-" * 50)
        print(f"[GRID] start evaluation for newly trained weights in experiment: {experiment_name_eval}")
        tenfold_evaluate(eval_cfg)
        print(f"[GRID] evaluation finished for experiment: {experiment_name_eval}")

    print("=" * 50)
    print("[INFO] Grid training & per-set evaluation finished. Running summary...")
    print("=" * 50)

    # すべてのセットの学習・評価が終わった後に、サマリを一度だけ作成
    # 元のcfgを変更せず、サマリ用の新しいcfgを作成
    summary_cfg = deepcopy(cfg)
    
    if getattr(summary_cfg, "evaluate", None) is None:
        summary_cfg.evaluate = Evaluate(run=None, tenfold=None, summary=None)

    # summary 評価の設定を作成（元のcfgは変更しない）
    grid_eval_summary = getattr(grid_eval_cfg, "summary", None) if grid_eval_cfg else None
    summary_cfg.evaluate.summary = create_eval_summary_config(
        grid_eval_summary_cfg=grid_eval_summary,
        experiment_name_eval=experiment_name_eval,
        search_space=ss,
    )

    try:
        summary_evaluate(summary_cfg)
    except Exception as e:
        print(f"[WARN] Summary evaluation failed: {e}")

    print("=" * 50)
    print("[INFO] Grid run finished (training + evaluation + summary).")
    print("=" * 50)
