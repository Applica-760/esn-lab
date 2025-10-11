# pyesn/setup/config_loader.py
import yaml
from pathlib import Path

def load_and_merge_configs(mode: str, variant: str) -> dict:

    base_yaml_path = Path("configs/base.yaml")
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {base_yaml_path}")
    
    yaml_path = Path("configs") / f"{mode}" / f"{variant}.yaml"
    with open(yaml_path, "r") as f:
        mode_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {yaml_path}")

    base_cfg.setdefault(mode, {})[variant] = mode_cfg
    
    return base_cfg