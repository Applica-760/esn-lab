# src/yourpkg/cli.py
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from mypkg.utils.config import (Config, TrainSingleCfg, TrainBatchCfg, TrainTenfoldCfg, TrainTenfoldSearchCfg,
                                PredictSingleCfg, PredictBatchCfg,
                                EvaluateRunCfg)
from mypkg.runner.train import single_train, batch_train
from mypkg.runner.train_10fold import tenfold_search_train
from mypkg.runner.predict import single_predict, batch_predict
from mypkg.runner.evaluate import single_evaluate


REGISTRY = {
    "train": {
        "variants": {
            "single": {"schema": TrainSingleCfg, "runner": single_train},
            "batch":  {"schema": TrainBatchCfg,  "runner": batch_train},
            # "tenfold": {"schema": TrainTenfoldCfg, "runner": tenfold_train}, 
            "tenfold_search": {"schema": TrainTenfoldSearchCfg, "runner": tenfold_search_train},
        }
    },
    "predict": {
        "variants": {
            "single": {"schema": PredictSingleCfg, "runner": single_predict},
            "batch":  {"schema": PredictBatchCfg, "runner": batch_predict}
        }
    },
    "evaluate": {
        "variants": {
            "run": {"schema": EvaluateRunCfg, "runner": single_evaluate},
            "cv" : {}   # cross validation
        }
    },
    "integ": {
        "variants": {
        }
    },
}

def main():
    # process args =================================================================================
    print("=====================================")
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", default=False,
                    help="デバッグ実行。runsに保存しない")

    sub = ap.add_subparsers(dest="mode", required=True)

    for mode, conf in REGISTRY.items():
        p = sub.add_parser(mode)
        variants = list(conf["variants"].keys())
        p.add_argument("variant", choices=variants)

    args = ap.parse_args()


    # load configs =================================================================================
    # load base config
    base_yaml_path = Path("configs/base.yaml")
    with open(base_yaml_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {base_yaml_path}")
    
    # load mode config
    yaml_path = Path("configs") / f"{args.mode}" / f"{args.variant}.yaml"
    with open(yaml_path, "r") as f:
        mode_cfg = yaml.safe_load(f)
    print(f"[OK] loaded config from {yaml_path}")

    # integ configs
    base_cfg[args.mode] = {args.variant : mode_cfg}
    run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{args.mode}-{getattr(args, 'variant', 'default')}"
    run_dir = Path("artifacts") / "runs" / run_name
    base_cfg["run_dir"] = str(run_dir)
    

    # initial setup =================================================================================
    if args.debug:
        run_dir = None
        print("[DEBUG MODE] runs ディレクトリは作りません")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = run_dir / "config.lock.yaml"
        with open(lock_path, "w") as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[OK] saved merged config to {lock_path}")

    print("=====================================")


    # exec runner =================================================================================
    regis = REGISTRY.get(args.mode, {}).get("variants", {}).get(args.variant)
    schema = OmegaConf.structured(Config)
    SchemaCls = regis.get("schema")

    if SchemaCls is not None:
        schema[args.mode][args.variant] = OmegaConf.structured(SchemaCls)
    cfg = OmegaConf.to_object(OmegaConf.merge(schema, base_cfg))

    runner = regis["runner"]
    print(f"[INFO] mode: {args.mode} is selected")
    print(f"[INFO] run {args.variant} {args.mode}")
    print(f"[ARTIFACT] run_dir={str(run_dir)}\n=====================================")
    runner(cfg)


if __name__ == "__main__":
    main()