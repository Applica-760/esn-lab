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

    run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{mode}-{variant}"
    run_dir = Path("artifacts") / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    lock_path = run_dir / "config.lock.yaml"
    with open(lock_path, "w") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] saved merged config to {lock_path}")
    
    return run_dir