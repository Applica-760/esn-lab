from esn_lab.model.esn import ESN
from esn_lab.optim.optim import Tikhonov
from projects.utils.data_loader import tenfold_data_loader
from projects.utils.config_loader import load_config
from projects.utils.grid_builder import build_param_grid
from projects.utils.weight_saver import save_tenfold_weights
from esn_lab.runner.train.tenfold_parallel import run_tenfold_parallel
# from esn_lab.runner.train.tenfold import run_tenfold

import time
"""
python -m projects.apps.tenfold_app --config projects/configs/tenfold_train.yaml
"""

def main():
    # configファイルの読み込み
    cfg = load_config()

    for i in range(10):

        # data loader データセットをロードするための主体
        data_folds, label_folds, id_folds = tenfold_data_loader(cfg.data_source[i])

        # grid builder パラメタサーチのデータを定義
        param_grid = build_param_grid(cfg)

        for params in param_grid:
            model = ESN(cfg.Nu, cfg.Ny, 
                        params["Nx"], params["density"], params["input_scale"], params["rho"])
            optimizer = Tikhonov(params["Nx"], cfg.Ny, 0.0)

            weights_list = run_tenfold_parallel(model, optimizer, data_folds, label_folds, cfg.workers)

            save_tenfold_weights(params, weights_list, cfg.output_dir[i])

            print(f"{params} is trained")

    return

if __name__ == "__main__":
    main()