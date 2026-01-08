import yaml
from types import SimpleNamespace


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    print("config loaded")
    return SimpleNamespace(**data)
