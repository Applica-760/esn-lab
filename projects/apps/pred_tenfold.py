import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from esn_lab.model.esn import ESN
from esn_lab.runner.pred.tenfold import pred_tenfold
from projects.utils.app_init import setup_app_environment, tenfold_data_loader
from projects.utils.weights import load_tenfold_weights, load_metadata, list_param_dirs
from projects.utils.prediction import save_pred_results, is_valid_result_file

"""
python -m projects.apps.pred_tenfold --config projects/configs/tenfold_pred.yaml
"""


def one_process(param_dir, mode, fold, data_folds, label_folds, id_folds, cfg, output_dir):
    """単一パラメータセットの10fold予測を実行"""
    # スキップ判定: 既に有効な結果ファイルが存在する場合はスキップ
    param_str = param_dir.name
    result_path = Path(output_dir) / fold / param_str / f"{mode}_results.json"
    if is_valid_result_file(str(result_path)):
        print(f"skipped (already exists): {param_str} fold={fold} mode={mode}")
        return
    
    params = load_metadata(param_dir)
    weights_list = load_tenfold_weights(param_dir)
    model = ESN(cfg.Nu, cfg.Ny, params["Nx"], params["density"], params["input_scale"], params["rho"])
    results = pred_tenfold(model, weights_list, data_folds, label_folds, id_folds, mode=mode)
    save_pred_results(results, str(result_path))
    print(f"proceed: {param_str} fold={fold} mode={mode}")
    return


def main():
    cfg, output_dir = setup_app_environment()
    weight_dir = Path(cfg.weight_dir)
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

    print("prediction is finished")
    return


if __name__ == "__main__":
    main()