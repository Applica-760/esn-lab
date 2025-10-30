# pyesn/setup/config_loader.py
import yaml
from pathlib import Path


def _canonical_mode(mode: str) -> str:
    """Map CLI mode to canonical config branch/dir names.

    - pred -> predict
    - eval -> evaluate
    - otherwise passthrough
    """
    return {"pred": "predict", "eval": "evaluate"}.get(mode, mode)

def load_and_merge_configs(mode: str, variant: str) -> dict:
    base_yaml_path = Path("configs/base.yaml")
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {base_yaml_path}")

    canon = _canonical_mode(mode)
    yaml_path = Path("configs") / canon / f"{variant}.yaml"
    with open(yaml_path, "r") as f:
        mode_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {yaml_path}")

    base_cfg.setdefault(canon, {})[variant] = mode_cfg

    return base_cfg