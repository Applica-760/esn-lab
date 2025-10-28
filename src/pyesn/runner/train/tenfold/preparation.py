from pathlib import Path

from . import utils
from pyesn.pipeline.tenfold_util import make_weight_filename, load_10fold_csv_mapping

def prepare_run_environment(cfg, tenfold_cfg=None):
    """
    実行に必要な設定を検証し、パスやファイルマッピングを準備する。

    引数:
        tenfold_cfg: TrainTenfoldCfg または同等のオブジェクト（csv_dir, weight_path を持つ）。
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

    # Prefer unified name 'weight_dir' with fallback to legacy 'weight_path'
    weight_dir_str = getattr(tenfold_cfg, "weight_dir", None) or getattr(tenfold_cfg, "weight_path", None)
    if not weight_dir_str:
        raise ValueError("Config requires 'weight_dir' (or legacy 'weight_path').")
    weight_dir = Path.cwd() / weight_dir_str
    weight_dir.mkdir(parents=True, exist_ok=True)
    
    csv_map = load_10fold_csv_mapping(csv_dir)
    letters = sorted(csv_map.keys())
    
    return {
        "weight_dir": weight_dir,
        "csv_map": csv_map,
        "letters": letters
    }

def determine_tasks_to_run(cfg, hp_overrides, letters, weight_dir):
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