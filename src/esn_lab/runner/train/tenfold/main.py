from pathlib import Path

from . import execution
from esn_lab.model.model_builder import get_model_param_str
from esn_lab.pipeline.data import create_data_loader_from_config
from esn_lab.utils.weight_management import WeightManager
from esn_lab.utils.execution_logging import ExecutionLogger


def _prepare_run_environment(cfg, tenfold_cfg=None):
    """
    実行に必要な設定を検証し、実行環境オブジェクトを準備する。

    Args:
        cfg: 設定オブジェクト
        tenfold_cfg: TrainTenfoldCfg または同等のオブジェクト。
                     None の場合は cfg.train.tenfold を参照。

    Returns:
        dict: 実行に必要な情報を含む辞書
            - data_loader: データローダーインスタンス
            - weight_manager: 重みファイル管理クラス
            - execution_logger: 実行ログ記録クラス
            - letters: 利用可能なfold ID一覧
    """
    tenfold_cfg = tenfold_cfg or getattr(getattr(cfg, "train", None), "tenfold", None)
    if tenfold_cfg is None:
        raise ValueError("Config 'cfg.train.tenfold' not found.")

    # experiment_name は必須
    experiment_name = getattr(tenfold_cfg, "experiment_name", None)
    if not experiment_name:
        raise ValueError("Config requires 'train.tenfold.experiment_name'.")
    
    print(f"[INFO] Using experiment: {experiment_name}")

    # 出力ディレクトリを設定
    experiment_dir = Path("outputs/experiments") / experiment_name
    experiment_dir = experiment_dir.expanduser().resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    weight_dir = experiment_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    # データローダーを生成
    data_loader, _ = create_data_loader_from_config(cfg, tenfold_cfg)
    letters = data_loader.get_available_folds()

    # ユーティリティクラスを初期化
    weight_manager = WeightManager(weight_dir)
    execution_logger = ExecutionLogger(experiment_dir)

    return {
        "data_loader": data_loader,
        "weight_manager": weight_manager,
        "execution_logger": execution_logger,
        "letters": letters,
    }


def _determine_skip_mode(tenfold_cfg):
    """設定からスキップモードを決定する。
    
    Args:
        tenfold_cfg: tenfold設定オブジェクト
    
    Returns:
        str: "never" | "if_exists" | "force_retrain"
    """
    skip_existing = getattr(tenfold_cfg, "skip_existing", True)
    force_retrain = getattr(tenfold_cfg, "force_retrain", False)
    
    if force_retrain:
        return "force_retrain"
    elif not skip_existing:
        return "never"
    else:
        return "if_exists"

def run_tenfold(cfg, *, overrides: dict | None = None, tenfold_cfg=None, parallel: bool | None = None, max_workers: int | None = None):
    """1パラメタ（=cfg.modelに対する上書き1セット）あたりの10-fold学習を実行する。

    - この関数は『単一のハイパーパラメタセット』に対して、未学習のfoldのみを学習する。
    - ハイパーパラメタの総当たりは上位の runner（integ/grid）が担当する。
    - overrides が None の場合は cfg.model の値をそのまま使用する。
    - 並列度は cfg.train.tenfold.workers に基づいて自動決定（1なら逐次、2以上で並列）。
    """
    # 1. 実行環境の準備
    env = _prepare_run_environment(cfg, tenfold_cfg=tenfold_cfg)
    
    # 有効なtenfold設定を取得
    ten_cfg_effective = tenfold_cfg or getattr(getattr(cfg, "train", None), "tenfold", None)
    
    # 2. スキップモードの決定
    skip_mode = _determine_skip_mode(ten_cfg_effective)
    
    # 3. 実行すべきタスク（fold）を決定
    hp_overrides = overrides or {}
    tasks_to_run = env["weight_manager"].determine_tasks_to_run(
        cfg, hp_overrides, env["letters"], skip_mode
    )

    if not tasks_to_run:
        print("[INFO] All folds for this parameter set are already trained. Nothing to do.")
        return

    # 4. 並列度の決定（明示指定がなければconfigから）
    auto_workers = int(getattr(ten_cfg_effective, "workers", 1) or 1)
    if parallel is None:
        parallel = auto_workers > 1
    if max_workers is None:
        max_workers = auto_workers

    # 5. タグ（実行記録CSVの識別用）
    hp_tag = get_model_param_str(cfg, overrides=hp_overrides)

    # 6. タスクを実行
    print("=" * 50)
    print(f"[INFO] Start tenfold training for a single param set: {hp_tag}")
    print("=" * 50)
    execution.execute_tasks(
        cfg=cfg,
        tenfold_cfg=ten_cfg_effective,
        data_loader=env["data_loader"],
        weight_manager=env["weight_manager"],
        execution_logger=env["execution_logger"],
        all_letters=env["letters"],
        hp_overrides=hp_overrides,
        tasks_to_run=tasks_to_run,
        parallel=parallel,
        max_workers=max_workers,
    )
    print("=" * 50)
    print("[INFO] Tenfold training finished for the parameter set above.")
    print("=" * 50)
