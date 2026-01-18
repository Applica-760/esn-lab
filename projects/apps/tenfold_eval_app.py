import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from esn_lab.model.esn import ESN
from esn_lab.runner.eval.tenfold import eval_tenfold
from projects.utils.config_loader import load_config
from projects.utils.data_loader import tenfold_data_loader
from projects.utils.weight_io import load_tenfold_weights, load_metadata, list_param_dirs
from projects.utils.result_saver import save_eval_results, is_valid_result_file

"""
python -m projects.apps.tenfold_eval_app --config projects/configs/tenfold_eval.yaml
"""


def one_process(param_dir, mode, fold, data_folds, label_folds, id_folds, cfg, output_dir):
    """単一パラメータセットの10fold評価を実行"""
    # スキップ判定: 既に有効な結果ファイルが存在する場合はスキップ
    param_str = param_dir.name
    result_path = Path(output_dir) / fold / param_str / f"{mode}_results.json"
    if is_valid_result_file(str(result_path)):
        print(f"skipped (already exists): {param_str} fold={fold} mode={mode}")
        return
    
    params = load_metadata(param_dir)
    weights_list = load_tenfold_weights(param_dir)
    model = ESN(cfg.Nu, cfg.Ny,
        params["Nx"], params["density"], params["input_scale"], params["rho"])
    results = eval_tenfold(model, weights_list,
        data_folds, label_folds, id_folds, mode=mode)
    save_eval_results(results, str(result_path))
    print(f"proceed: {param_str} fold={fold} mode={mode}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # 1. configの読み込み
    cfg = load_config(args.config)
    weight_dir = Path(cfg.weight_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "eval_config.lock.yaml")
    modes = cfg.mode  # リスト形式

    # 3. modeループ
    for mode in modes:
        for fold in cfg.folds:
            print(f"mode={mode} fold={fold}")
            data_source = Path(cfg.data_source_base_dir) / fold
            fold_dir = weight_dir / fold

            data_folds, label_folds, id_folds = tenfold_data_loader(data_source)
            param_dirs = list_param_dirs(fold_dir)       # パラメータディレクトリ一覧を取得

            with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                futures = [
                    executor.submit(one_process, param_dir, mode, fold, data_folds, label_folds, id_folds, cfg, output_dir)
                    for param_dir in param_dirs
                ]
                for future in futures:
                    future.result()

    print("eval is finished")
    return


if __name__ == "__main__":
    main()