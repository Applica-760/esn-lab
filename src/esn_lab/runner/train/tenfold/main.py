from pathlib import Path

from . import execution
from esn_lab.model.model_builder import get_model_param_str
from esn_lab.pipeline.tenfold_util import make_weight_filename, load_10fold_csv_mapping


def _prepare_run_environment(cfg, tenfold_cfg=None):
    """
    実行に必要な設定を検証し、パスやファイルマッピングを準備する。

    引数:
    tenfold_cfg: TrainTenfoldCfg または同等のオブジェクト（csv_dir, weight_dir を持つ）。
                     None の場合は cfg.train.tenfold を参照。

    戻り値:
        dict: 実行に必要な情報（weight_dir, csv_map, letters）を含む辞書
    """
    tenfold_cfg = tenfold_cfg or getattr(getattr(cfg, "train", None), "tenfold", None)
    if tenfold_cfg is None:
        raise ValueError("Config 'cfg.train.tenfold' not found.")

    csv_dir = Path(getattr(tenfold_cfg, "csv_dir")).expanduser().resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")

    # Require unified name 'weight_dir'
    weight_dir_str = getattr(tenfold_cfg, "weight_dir", None)
    if not weight_dir_str:
        raise ValueError("Config requires 'weight_dir'.")
    weight_dir = Path.cwd() / weight_dir_str
    weight_dir.mkdir(parents=True, exist_ok=True)

    csv_map = load_10fold_csv_mapping(csv_dir)
    letters = sorted(csv_map.keys())

    return {
        "weight_dir": weight_dir,
        "csv_map": csv_map,
        "letters": letters,
    }


def _determine_tasks_to_run(cfg, hp_overrides, letters, weight_dir):
    """
    特定のハイパーパラメータに対し、未実行のfold（タスク）を特定する。
    重みファイルが既に存在する場合はスキップ対象とする。

    戻り値:
        list[str]: 実行すべき 'leave-out-letter' のリスト
    """
    tasks_to_run = []
    for leave in letters:
        train_letters = [x for x in letters if x != leave]
        tag = "".join(train_letters)
        weight_filename = make_weight_filename(cfg=cfg, overrides=hp_overrides, train_tag=tag)
        expected_path = weight_dir / weight_filename

        if expected_path.exists():
            print(f"[SKIP] Weight file found, skipping fold '{leave}': {expected_path.name}")
        else:
            tasks_to_run.append(leave)

    return tasks_to_run

def run_tenfold(cfg, *, overrides: dict | None = None, tenfold_cfg=None, parallel: bool | None = None, max_workers: int | None = None):
    """1パラメタ（=cfg.modelに対する上書き1セット）あたりの10-fold学習を実行する。

    - この関数は『単一のハイパーパラメタセット』に対して、未学習のfoldのみを学習する。
    - ハイパーパラメタの総当たりは上位の runner（integ/grid）が担当する。
    - overrides が None の場合は cfg.model の値をそのまま使用する。
    - 並列度は cfg.train.tenfold.workers に基づいて自動決定（1なら逐次、2以上で並列）。
    """
    # 1. 実行環境の準備（integ.grid から渡された tenfold 設定があればそれを使う）
    env = _prepare_run_environment(cfg, tenfold_cfg=tenfold_cfg)

    # 2. 実行すべきタスク（fold）を決定（本ランナーは単一パラメタセットのみ扱う）
    hp_overrides = overrides or {}
    tasks_to_run = _determine_tasks_to_run(
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
