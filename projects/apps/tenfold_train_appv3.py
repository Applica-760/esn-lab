import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from esn_lab.model.esn import ESN
from esn_lab.optim.optim import Tikhonov
from esn_lab.utils.fold_splitter import get_train_folds
from esn_lab.pipeline.train.trainer import train
from projects.utils.data_loader import tenfold_data_loader
from projects.utils.config_loader import load_config
from projects.utils.grid_builder import build_param_grid
from projects.utils.weight_io import save_single_weight


"""
python -m projects.apps.tenfold_train_appv3 --config projects/configs/tenfold_train.yaml
"""

def one_process(params, fold_idx, data_folds, label_folds, id_folds, Nu, Ny, output_dir):
    """単一のfold_idx + paramsの組み合わせに対する学習処理"""
    U_list, D_list, _ = get_train_folds(data_folds, label_folds, id_folds, fold_idx)
    model = ESN(Nu, Ny, params["Nx"], params["density"], params["input_scale"], params["rho"])
    optimizer = Tikhonov(params["Nx"], Ny, 0.0)
    output_weight = train(model, optimizer, U_list, D_list)
    save_single_weight(params, output_weight, fold_idx, output_dir)
    print(f"proceed: {params} {fold_idx}")
    return


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
        print(f"fold {fold}")
        data_source = Path(cfg.data_source_base_dir) / fold
        output_dir = Path(cfg.output_base_dir) / fold

        # data loader データセットをロードするための主体
        data_folds, label_folds, id_folds = tenfold_data_loader(data_source)
        # grid builder パラメタサーチのデータを定義
        param_grid = build_param_grid(cfg)
        # param_grid × range(10) をフラットに展開
        jobs = [(params, i) for params in param_grid for i in range(10)]

        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = [
                executor.submit(one_process, params, i, data_folds, label_folds, id_folds, cfg.Nu, cfg.Ny, output_dir)
                for params, i in jobs
            ]
            for future in futures:
                future.result()

    print("train is finished")
    return

if __name__ == "__main__":
    main()