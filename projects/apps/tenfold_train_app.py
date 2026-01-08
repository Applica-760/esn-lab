import argparse
import shutil
from pathlib import Path
from esn_lab.model.esn import ESN
from esn_lab.optim.optim import Tikhonov
from projects.utils.data_loader import tenfold_data_loader
from projects.utils.config_loader import load_config
from projects.utils.grid_builder import build_param_grid
from projects.utils.weight_saver import save_tenfold_weights
from esn_lab.runner.train.tenfold_parallel import run_tenfold_parallel
# from esn_lab.runner.train.tenfold import run_tenfold

"""
python -m projects.apps.tenfold_train_app --config projects/configs/tenfold_train.yaml
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # configファイルの読み込み
    cfg = load_config(args.config)
    
    # output_base_dirの作成とyaml.lockの保存
    output_dir = Path(cfg.output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = output_dir / "config.lock.yaml"
    shutil.copy(args.config, lock_file)

    for fold in cfg.folds:
        data_source = Path(cfg.data_source_base_dir) / fold
        output_dir = Path(cfg.output_base_dir) / fold

        # data loader データセットをロードするための主体
        data_folds, label_folds, id_folds = tenfold_data_loader(data_source)

        # grid builder パラメタサーチのデータを定義
        param_grid = build_param_grid(cfg)

        for params in param_grid:
            model = ESN(cfg.Nu, cfg.Ny, 
                        params["Nx"], params["density"], params["input_scale"], params["rho"])
            optimizer = Tikhonov(params["Nx"], cfg.Ny, 0.0)

            weights_list = run_tenfold_parallel(model, optimizer, data_folds, label_folds, cfg.workers)

            save_tenfold_weights(params, weights_list, output_dir)

            print(f"{params} is trained")

    return

if __name__ == "__main__":
    main()