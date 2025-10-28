from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path

from pyesn.runner.train.tenfold.main import run_tenfold
from pyesn.runner.train.tenfold.utils import flatten_search_space
from pyesn.runner.eval.evaluate import tenfold_evaluate, summary_evaluate
from pyesn.setup.config import Evaluate, EvaluateTenfoldCfg, EvaluateSummaryCfg


def run_grid(cfg) -> None:
    """ハイパーパラメタグリッドを総当たりし、各組み合わせでシンプルな tenfold 学習を実行する。

    期待する設定（cfg.integ.grid.train）:
      - csv_dir: 10-fold分割済みCSVのディレクトリ
    - weight_dir: 重み出力先ディレクトリ（従来 'weight_path' は後方互換）
      - workers: 並列ワーカ数（1で逐次、2以上で並列）
      - search_space: {"model.<field>": [values, ...], ...}
    """
    integ = getattr(cfg, "integ", None)
    if integ is None or getattr(integ, "grid", None) is None:
        raise ValueError("Config 'integ.grid' not found.")

    grid_cfg = getattr(integ, "grid")
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
        # Require unified 'weight_dir'
        weight_dir_str: str = (
            getattr(grid_eval_cfg.tenfold, "weight_dir", None)
            or getattr(train_cfg, "weight_dir", None)
        )
        eval_workers: int = int(getattr(grid_eval_cfg.tenfold, "workers", None) or getattr(train_cfg, "workers", 1) or 1)
        eval_parallel: bool = bool(getattr(grid_eval_cfg.tenfold, "parallel", True))
    else:
        csv_dir_str = getattr(train_cfg, "csv_dir")
        weight_dir_str = getattr(train_cfg, "weight_dir", None)
        eval_workers = auto_workers
        eval_parallel = True
    if not weight_dir_str:
        raise ValueError("'weight_dir' is required in integ.grid.eval.tenfold or integ.grid.train.")

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

        # 学習直後に、未評価の重みのみ tenfold 評価を実行する
        # tenfold 評価の設定を優先順位で決定
        if getattr(cfg, "evaluate", None) is None:
            cfg.evaluate = Evaluate(run=None, tenfold=None, summary=None)
        # integ.grid.eval.tenfold があればそれを使用、なければ補完値
        if grid_eval_cfg and getattr(grid_eval_cfg, "tenfold", None):
            cfg.evaluate.tenfold = grid_eval_cfg.tenfold
        else:
            cfg.evaluate.tenfold = EvaluateTenfoldCfg(
                csv_dir=csv_dir_str,
                weight_dir=weight_dir_str,
                workers=eval_workers,
                parallel=eval_parallel,
            )
        print("-" * 50)
        print(f"[GRID] start evaluation for newly trained weights in: {weight_dir_str}")
        tenfold_evaluate(cfg)
        print(f"[GRID] evaluation finished for: {weight_dir_str}")

    print("=" * 50)
    print("[INFO] Grid training & per-set evaluation finished. Running summary...")
    print("=" * 50)

    # すべてのセットの学習・評価が終わった後に、サマリを一度だけ作成
    # integ.grid.eval.summary があればそれを使用、なければ簡易設定で補完
    if getattr(cfg, "evaluate", None) is None:
        cfg.evaluate = Evaluate(run=None, tenfold=None, summary=None)

    if grid_eval_cfg and getattr(grid_eval_cfg, "summary", None):
        # ユーザ指定を尊重しつつ、vary_values が未指定なら search_space から自動補完
        cfg.evaluate.summary = grid_eval_cfg.summary
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
            weight_dir=weight_dir_str,
            vary_param=vary_param,
        )

    try:
        summary_evaluate(cfg)
    except Exception as e:
        print(f"[WARN] Summary evaluation failed: {e}")

    print("=" * 50)
    print("[INFO] Grid run finished (training + evaluation + summary).")
    print("=" * 50)
