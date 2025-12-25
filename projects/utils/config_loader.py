import argparse
import yaml
from types import SimpleNamespace


def load_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        data = yaml.safe_load(f)
    print("config loaded")
    return SimpleNamespace(**data)
