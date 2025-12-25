# esn_lab/setup/args.py
import argparse
from .registry import REGISTRY

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", default=False,
                    help="デバッグ実行。runsに保存しない")

    sub = ap.add_subparsers(dest="mode", required=True)

    sub.add_parser("init", help="configsディレクトリを初期化します。")

    for mode, conf in REGISTRY.items():
        p = sub.add_parser(mode)
        variants = list(conf["variants"].keys())
        p.add_argument("variant", choices=variants)

    args = ap.parse_args()
    return args