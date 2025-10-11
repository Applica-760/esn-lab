# import argparse
# import yaml
# from pathlib import Path
# from omegaconf import OmegaConf

# from pyesn.setup.workspace import initialize_configs, setup_rundir
# # scheme
# from pyesn.setup.config import (Config, TrainSingleCfg, TrainBatchCfg, TrainTenfoldSearchCfg,
#                                 PredictSingleCfg, PredictBatchCfg,
#                                 EvaluateRunCfg)
# # runner
# from pyesn.runner.train import single_train, batch_train
# from pyesn.runner.train_10fold import tenfold_search_train
# from pyesn.runner.predict import single_predict, batch_predict
# from pyesn.runner.evaluate import single_evaluate



# REGISTRY = {
#     "train": {
#         "variants": {
#             "single": {"schema": TrainSingleCfg, "runner": single_train},
#             "batch":  {"schema": TrainBatchCfg,  "runner": batch_train},
#             "tenfold_search": {"schema": TrainTenfoldSearchCfg, "runner": tenfold_search_train},
#         }
#     },
#     "predict": {
#         "variants": {
#             "single": {"schema": PredictSingleCfg, "runner": single_predict},
#             "batch":  {"schema": PredictBatchCfg, "runner": batch_predict}
#         }
#     },
#     "evaluate": {
#         "variants": {
#             "run": {"schema": EvaluateRunCfg, "runner": single_evaluate},
#             "cv" : {}
#         }
#     },
#     "integ": {
#         "variants": {}
#     },
# }


# def main():
#     print("=====================================")
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--debug", action="store_true", default=False,
#                     help="デバッグ実行。runsに保存しない")

#     sub = ap.add_subparsers(dest="mode", required=True)

#     sub.add_parser("init", help="configsディレクトリを初期化します。")

#     for mode, conf in REGISTRY.items():
#         p = sub.add_parser(mode)
#         variants = list(conf["variants"].keys())
#         p.add_argument("variant", choices=variants)

#     args = ap.parse_args()

#     if args.mode == "init":
#         initialize_configs()
#         return

#     # --- config loading ---
#     base_yaml_path = Path("configs/base.yaml")
#     with open(base_yaml_path, "r") as f:
#         base_cfg = yaml.safe_load(f)
#     print(f"[OK] loaded config from {base_yaml_path}")
    
#     yaml_path = Path("configs") / f"{args.mode}" / f"{args.variant}.yaml"
#     with open(yaml_path, "r") as f:
#         mode_cfg = yaml.safe_load(f)
#     print(f"[OK] loaded config from {yaml_path}")

#     base_cfg.setdefault(args.mode, {})[args.variant] = mode_cfg
    
#     # --- run_dir setup ---
#     run_dir = setup_rundir(args.mode, args.variant, args.debug, base_cfg)
#     if run_dir:
#         base_cfg["run_dir"] = str(run_dir)
#     print("=====================================")

#     # --- runner execution ---
#     regis = REGISTRY.get(args.mode, {}).get("variants", {}).get(args.variant)
#     if not regis or "runner" not in regis:
#         print(f"[ERROR] Runner for {args.mode}/{args.variant} is not registered.")
#         return
        
#     schema = OmegaConf.structured(Config)
#     SchemaCls = regis.get("schema")
#     if SchemaCls:
#         schema.setdefault(args.mode, {})[args.variant] = OmegaConf.structured(SchemaCls)
        
#     cfg = OmegaConf.to_object(OmegaConf.merge(schema, base_cfg))

#     runner = regis["runner"]
#     print(f"[INFO] mode: {args.mode} is selected")
#     print(f"[INFO] run {args.variant} {args.mode}")
#     if run_dir:
#         print(f"[ARTIFACT] run_dir={str(run_dir)}")
#     print("=====================================")
#     runner(cfg)


# if __name__ == "__main__":
#     main()



# pyesn/cli.py
from pyesn.setup.workspace import initialize_configs, setup_rundir
from pyesn.setup.args import parse_args
from pyesn.setup.config_loader import load_and_merge_configs
from pyesn.setup.executor import execute_runner

def main():
    print("=====================================")
    args = parse_args()

    # モードに応じた処理の振り分け
    if args.mode == "init":
        initialize_configs()
        return

    # 1. 設定ファイルの読み込みとマージ
    merged_cfg = load_and_merge_configs(args.mode, args.variant)
    
    # 2. 実行ディレクトリのセットアップ
    run_dir = setup_rundir(args.mode, args.variant, args.debug, merged_cfg)
    print("=====================================")
    
    # 3. ランナーの実行
    execute_runner(args, merged_cfg, run_dir)


if __name__ == "__main__":
    main()