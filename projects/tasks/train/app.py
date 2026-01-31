from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from esn_lab.model.esn import ESN
from esn_lab.optim.optim import Tikhonov
from esn_lab.utils.fold_splitter import get_train_folds
from esn_lab.pipeline.train.trainer import train
from projects.utils.app_init import tenfold_data_loader, build_param_grid
from projects.utils.weights import save_single_weight, build_param_str, is_valid_weight_file


def one_process(params, fold_idx, data_folds, label_folds, id_folds, Nu, Ny, output_dir):
    """単一のfold_idx + paramsの組み合わせに対する学習処理"""
    # スキップ判定: 既に有効な重みファイルが存在する場合はスキップ
    param_str = build_param_str(params)
    weight_path = Path(output_dir) / param_str / f"fold{fold_idx}.npz"
    if is_valid_weight_file(str(weight_path)):
        print(f"skipped (already exists): {params} fold_idx={fold_idx}")
        return
    
    U_list, D_list, _ = get_train_folds(data_folds, label_folds, id_folds, fold_idx)
    model = ESN(Nu, Ny, params["Nx"], params["density"], params["input_scale"], params["rho"])
    optimizer = Tikhonov(params["Nx"], Ny, 0.0)
    output_weight = train(model, optimizer, U_list, D_list)
    save_single_weight(params, output_weight, fold_idx, output_dir)
    print(f"proceed: {params} {fold_idx}")
    return


def main(cfg):
    
    for group in cfg.groups:
        print(f"group {group}")
        data_source = Path(cfg.data_source_base_dir) / group
        group_output_dir = cfg.output_dir / group

        data_folds, label_folds, id_folds = tenfold_data_loader(data_source)    # data loader データセットをロードするための主体
        param_grid = build_param_grid(cfg)                                   # grid builder パラメタサーチのデータを定義
        jobs = [(params, i) for params in param_grid for i in range(10)]    # param_grid × range(10) をフラットに展開

        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = [
                executor.submit(one_process, params, i, data_folds, label_folds, id_folds, cfg.Nu, cfg.Ny, group_output_dir)
                for params, i in jobs
            ]
            for future in futures:
                future.result()

    print("train is finished")
    return