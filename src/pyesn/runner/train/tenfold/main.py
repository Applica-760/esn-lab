from . import preparation
from . import execution
from pyesn.model.model_builder import get_model_param_str

def run_tenfold(cfg, *, overrides: dict | None = None, tenfold_cfg=None, parallel: bool | None = None, max_workers: int | None = None):
    """1パラメタ（=cfg.modelに対する上書き1セット）あたりの10-fold学習を実行する。

    - この関数は『単一のハイパーパラメタセット』に対して、未学習のfoldのみを学習する。
    - ハイパーパラメタの総当たりは上位の runner（integ/grid）が担当する。
    - overrides が None の場合は cfg.model の値をそのまま使用する。
    - 並列度は cfg.train.tenfold.workers に基づいて自動決定（1なら逐次、2以上で並列）。
    """
    # 1. 実行環境の準備（integ.grid から渡された tenfold 設定があればそれを使う）
    env = preparation.prepare_run_environment(cfg, tenfold_cfg=tenfold_cfg)

    # 2. 実行すべきタスク（fold）を決定（本ランナーは単一パラメタセットのみ扱う）
    hp_overrides = overrides or {}
    tasks_to_run = preparation.determine_tasks_to_run(
        cfg, hp_overrides, env["letters"], env["weight_dir"]
    )

    if not tasks_to_run:
        print("[INFO] All folds for this parameter set are already trained. Nothing to do.")
        return

    # 3. 並列度の決定（明示指定がなければconfigから）
    ten_cfg_effective = tenfold_cfg or getattr(getattr(cfg, "train", None), "tenfold", None)
    auto_workers = int(getattr(ten_cfg_effective, "workers", 1) or 1)
    if parallel is None:
        parallel = auto_workers > 1
    if max_workers is None:
        max_workers = auto_workers

    # 4. タグ（実行記録CSVの識別用）
    hp_tag = get_model_param_str(cfg, overrides=hp_overrides)

    # 5. タスクを実行
    print("=" * 50)
    print(f"[INFO] Start tenfold training for a single param set: {hp_tag}")
    print("=" * 50)
    execution.execute_tasks(
        cfg, env, hp_overrides, hp_tag, tasks_to_run, parallel, max_workers
    )
    print("=" * 50)
    print("[INFO] Tenfold training finished for the parameter set above.")
    print("=" * 50)

