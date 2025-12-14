import shutil
from pathlib import Path
from datetime import datetime
import yaml

# 設定ファイルの初期化
def initialize_configs():
    source_dir = Path(__file__).parent.parent / "config_templates"
    dest_dir = Path.cwd() / "configs"
    
    if dest_dir.exists():
        print(f"[INFO] '{dest_dir}' は既に存在するため、初期化をスキップします。")
        return
    try:
        shutil.copytree(source_dir, dest_dir)
        print(f"[OK] 設定ファイルが '{dest_dir}' にコピーされました。")
    except Exception as e:
        print(f"[ERROR] 設定ファイルのコピーに失敗しました: {e}")


# 実行ディレクトリのセットアップ
def setup_rundir(mode: str, variant: str, debug: bool, merged_cfg: dict) -> Path | None:
    if debug:
        print("[DEBUG MODE] runs ディレクトリは作りません")
        return None

    # experiment_name が指定されている場合は experiments/ 配下に配置
    experiment_name = _extract_experiment_name(mode, variant, merged_cfg)
    
    if experiment_name:
        run_dir = Path("outputs") / "experiments" / experiment_name
        print(f"[INFO] Using experiment: {experiment_name}")
    else:
        # 通常の単発実験は runs/ 配下にタイムスタンプ付きで配置
        run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{mode}-{variant}"
        run_dir = Path("outputs") / "runs" / run_name
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    lock_path = run_dir / "config.lock.yaml"
    with open(lock_path, "w") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] saved merged config to {lock_path}")
    
    return run_dir


def _extract_experiment_name(mode: str, variant: str, merged_cfg: dict) -> str | None:
    """設定から experiment_name を抽出する。
    
    tenfold, grid などの長時間実験用モードでは experiment_name の使用を推奨。
    single, batch などの単発実験では None を返す。
    """
    # モードとバリアントに応じた設定パスを確認
    canonical_mode = {"pred": "predict", "eval": "evaluate"}.get(mode, mode)
    
    # train.tenfold, evaluate.tenfold, integ.grid などから experiment_name を探す
    if canonical_mode == "train" and variant == "tenfold":
        return merged_cfg.get("train", {}).get("tenfold", {}).get("experiment_name")
    elif canonical_mode == "evaluate" and variant in ["tenfold", "summary", "analysis"]:
        eval_cfg = merged_cfg.get("evaluate", {})
        return eval_cfg.get(variant, {}).get("experiment_name")
    elif canonical_mode == "integ" and variant == "grid":
        return merged_cfg.get("integ", {}).get("grid", {}).get("experiment_name")
    
    return None