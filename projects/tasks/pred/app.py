from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from esn_lab.model.esn import ESN
from esn_lab.runner.pred.tenfold import pred_tenfold
from projects.utils.app_init import tenfold_data_loader
from projects.utils.weights import load_tenfold_weights, load_metadata, list_param_dirs
from projects.utils.prediction import save_pred_results, is_valid_result_file

def one_process(param_dir, mode, group, data_folds, label_folds, id_folds, cfg, output_dir):
    """単一パラメータセットの10fold予測を実行"""
    # スキップ判定: 既に有効な結果ファイルが存在する場合はスキップ
    param_str = param_dir.name
    result_path = Path(output_dir) / group / param_str / f"{mode}_results.json"
    if is_valid_result_file(str(result_path)):
        print(f"skipped (already exists): {param_str} group={group} mode={mode}")
        return
    
    params = load_metadata(param_dir)
    weights_list = load_tenfold_weights(param_dir)
    model = ESN(cfg.Nu, cfg.Ny, params["Nx"], params["density"], params["input_scale"], params["rho"])
    results = pred_tenfold(model, weights_list, data_folds, label_folds, id_folds, mode=mode)
    save_pred_results(results, str(result_path))
    print(f"proceed: {param_str} group={group} mode={mode}")
    return

def main(cfg):
    """
    予測タスクのメイン関数
    
    Args:
        cfg: 設定オブジェクト（cfg.output_dirに出力ディレクトリが含まれる）
    """
    weight_dir = Path(cfg.weight_dir)
    modes = cfg.mode  # リスト形式

    # 3. modeループ
    for mode in modes:
        for group in cfg.groups:
            print(f"mode={mode} group={group}")
            data_source = Path(cfg.data_source_base_dir) / group
            group_dir = weight_dir / group

            data_folds, label_folds, id_folds = tenfold_data_loader(data_source)
            param_dirs = list_param_dirs(group_dir)       # パラメータディレクトリ一覧を取得

            with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
                futures = [
                    executor.submit(one_process, param_dir, mode, group, data_folds, label_folds, id_folds, cfg, cfg.output_dir)
                    for param_dir in param_dirs
                ]
                for future in futures:
                    future.result()

    print("prediction is finished")
    return