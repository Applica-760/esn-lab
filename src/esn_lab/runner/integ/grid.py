from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path

from esn_lab.runner.train.tenfold.main import run_tenfold
from esn_lab.utils.param_grid import flatten_search_space
from esn_lab.runner.eval.evaluate import tenfold_evaluate, summary_evaluate
from esn_lab.setup.config import (
    DataSourceCfg,
    Evaluate,
    EvaluateTenfoldCfg,
    EvaluateSummaryCfg,
    TrainTenfoldCfg,
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

    # helper to read attribute or dict key
    def _g(obj, key, default=None):
        if obj is None:
            return default
        try:
            return getattr(obj, key)
        except Exception:
            try:
                return obj.get(key, default)  # type: ignore[attr-defined]
            except Exception:
                return default

    # 新形式: integ.grid が単一の grid.yaml 形式を持つ
    # structure: base, param_grid (or search_space), train, eval, summary
    is_new_grid_schema = (_g(grid_cfg, "param_grid") is not None) or (_g(grid_cfg, "base") is not None)

    if is_new_grid_schema:
        base = _g(grid_cfg, "base") or {}
        param_grid = _g(grid_cfg, "param_grid") or _g(grid_cfg, "search_space")
        train_template = _g(grid_cfg, "train") or {}
        grid_eval_cfg = _g(grid_cfg, "eval") or {}
        summary_template = _g(grid_cfg, "summary") or {}

        # 組み合わせ展開
        ss = param_grid
        combos: List[Tuple[Dict, str]] = flatten_search_space(ss)

        # experiment_name は必須
        experiment_name = _g(grid_cfg, "experiment_name") or _g(train_template, "experiment_name") or _g(base, "experiment_name")
        if not experiment_name:
            raise ValueError("integ.grid requires 'experiment_name'.")
        
        print(f"[INFO] Using experiment: {experiment_name}")

        # tenfold 設定を base + train_template から作る
        csv_dir = _g(train_template, "csv_dir") or _g(base, "csv_dir")
        data_source = _g(train_template, "data_source") or _g(base, "data_source")
        
        # data_source が辞書の場合は DataSourceCfg オブジェクトに変換
        if data_source is not None and isinstance(data_source, dict):
            data_source = DataSourceCfg(
                type=data_source.get("type", "csv"),
                csv_dir=data_source.get("csv_dir"),
                npy_dir=data_source.get("npy_dir")
            )
        
        workers = _g(train_template, "workers") or _g(base, "workers") or 1
        skip_existing = _g(train_template, "skip_existing")
        if skip_existing is None:
            skip_existing = _g(base, "skip_existing")
        if skip_existing is None:
            skip_existing = True
        force_retrain = _g(train_template, "force_retrain") or _g(base, "force_retrain") or False

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
    
    # data_source_eval が辞書の場合は DataSourceCfg オブジェクトに変換
    if data_source_eval is not None and isinstance(data_source_eval, dict):
        data_source_eval = DataSourceCfg(
            type=data_source_eval.get("type", "csv"),
            csv_dir=data_source_eval.get("csv_dir"),
            npy_dir=data_source_eval.get("npy_dir")
        )
    
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
        # tenfold 評価の設定を優先順位で決定
        if getattr(cfg, "evaluate", None) is None:
            cfg.evaluate = Evaluate(run=None, tenfold=None, summary=None)
        # integ.grid.eval.tenfold があればそれを使用、なければ補完値
        if grid_eval_cfg and getattr(grid_eval_cfg, "tenfold", None):
            # ユーザ指定の tenfold をベースに、不足項目を補完
            cfg.evaluate.tenfold = grid_eval_cfg.tenfold
            if getattr(cfg.evaluate.tenfold, "csv_dir", None) in (None, ""):
                cfg.evaluate.tenfold.csv_dir = csv_dir_str
            if getattr(cfg.evaluate.tenfold, "data_source", None) is None:
                cfg.evaluate.tenfold.data_source = data_source_eval
            if getattr(cfg.evaluate.tenfold, "experiment_name", None) in (None, ""):
                cfg.evaluate.tenfold.experiment_name = experiment_name_eval
            if getattr(cfg.evaluate.tenfold, "workers", None) in (None, 0):
                cfg.evaluate.tenfold.workers = eval_workers
            if getattr(cfg.evaluate.tenfold, "parallel", None) is None:
                cfg.evaluate.tenfold.parallel = eval_parallel
        else:
            cfg.evaluate.tenfold = EvaluateTenfoldCfg(
                csv_dir=csv_dir_str,
                data_source=data_source_eval,
                experiment_name=experiment_name_eval,
                workers=eval_workers,
                parallel=eval_parallel,
            )
        # このセットだけを評価対象にするため search_space を1要素で付与
        # overrides は {"Nx":..,"density":..} 形式。search_space は "model." 接頭を要求。
        one_search = {f"model.{k}": [v] for k, v in (overrides or {}).items()}
        cfg.evaluate.tenfold.search_space = one_search if one_search else None
        print("-" * 50)
        print(f"[GRID] start evaluation for newly trained weights in experiment: {experiment_name_eval}")
        tenfold_evaluate(cfg)
        print(f"[GRID] evaluation finished for experiment: {experiment_name_eval}")

    print("=" * 50)
    print("[INFO] Grid training & per-set evaluation finished. Running summary...")
    print("=" * 50)

    # すべてのセットの学習・評価が終わった後に、サマリを一度だけ作成
    # integ.grid.eval.summary があればそれを使用、なければ簡易設定で補完
    if getattr(cfg, "evaluate", None) is None:
        cfg.evaluate = Evaluate(run=None, tenfold=None, summary=None)

    if grid_eval_cfg and getattr(grid_eval_cfg, "summary", None):
        # ユーザ指定を尊重しつつ、不足項目を補完
        cfg.evaluate.summary = grid_eval_cfg.summary
        try:
            if getattr(cfg.evaluate.summary, "experiment_name", None) in (None, ""):
                cfg.evaluate.summary.experiment_name = experiment_name_eval
        except Exception:
            pass
        try:
            if getattr(cfg.evaluate.summary, "vary_values", None) in (None, []) and ss:
                vary_param = getattr(cfg.evaluate.summary, "vary_param", None) or "Nx"
                # search_space は 'model.X' か 'X' で指定されうる
                candidates = []
                if isinstance(ss, dict):
                    if f"model.{vary_param}" in ss:
                        candidates = list(ss[f"model.{vary_param}"])
                    elif vary_param in ss:
                        candidates = list(ss[vary_param])
                if candidates:
                    cfg.evaluate.summary.vary_values = candidates
        except Exception as e:
            print(f"[WARN] Failed to auto-fill summary.vary_values: {e}")
    else:
        # vary_param は search_space が単一項目ならそれを用いる
        vary_param = "Nx"
        try:
            if isinstance(ss, dict) and len(ss) >= 1:
                fields = []
                for k in ss.keys():
                    if isinstance(k, str) and k.startswith("model."):
                        fields.append(k.split(".", 1)[1])
                if len(fields) == 1:
                    vary_param = fields[0]
                elif "Nx" in fields:
                    vary_param = "Nx"
                elif len(fields) > 1:
                    vary_param = fields[0]
        except Exception:
            pass
        cfg.evaluate.summary = EvaluateSummaryCfg(
            experiment_name=experiment_name_eval,
            vary_param=vary_param,
        )

    try:
        summary_evaluate(cfg)
    except Exception as e:
        print(f"[WARN] Summary evaluation failed: {e}")

    print("=" * 50)
    print("[INFO] Grid run finished (training + evaluation + summary).")
    print("=" * 50)
