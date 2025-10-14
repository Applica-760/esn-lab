from . import utils
from . import preparation
from . import execution

def run_tenfold(cfg, *, parallel: bool = True, max_workers: int = 10):
    # 1. 実行環境の準備
    env = preparation.prepare_run_environment(cfg)
    hp_combos = utils.flatten_search_space(getattr(cfg.train.tenfold, "search_space", None))

    # 2. 各ハイパーパラメータの組み合わせについてループ
    for hp_overrides, hp_tag in hp_combos:
        print("="*50)
        print(f"[INFO] Processing hyperparameter combo: {hp_tag}")
        print("="*50)

        # 3. 実行すべきタスク（fold）を決定
        tasks_to_run = preparation.determine_tasks_to_run(
            cfg, hp_overrides, env["letters"], env["weight_dir"]
        )

        if not tasks_to_run:
            print("[INFO] All folds for this combo are already trained. Skipping.")
            continue
        
        # 4. タスクを実行
        execution.execute_tasks(
            cfg, env, hp_overrides, hp_tag, tasks_to_run, parallel, max_workers
        )

    print("="*50)
    print("[INFO] 10-fold hyperparameter search finished.")
    print("="*50)