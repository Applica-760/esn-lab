import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from esn_lab.model.esn import ESN
from esn_lab.runner.eval.tenfold import eval_tenfold
from projects.utils.config_loader import load_config
from projects.utils.data_loader import tenfold_data_loader
from projects.utils.weight_io import load_tenfold_weights, load_metadata, list_param_dirs
from projects.utils.result_saver import save_eval_results

"""
python -m projects.apps.tenfold_eval_app --config projects/configs/tenfold_eval.yaml
"""


def eval_single_param(train_cfg, param_dir, data_folds, label_folds, id_folds, mode):
    """単一パラメータセットの10fold評価を実行"""
    # メタデータと重みをロード
    metadata = load_metadata(param_dir)
    weights_list = load_tenfold_weights(param_dir)
    model = ESN(train_cfg.Nu, train_cfg.Ny,
        metadata["Nx"], metadata["density"], metadata["input_scale"], metadata["rho"])
    results = eval_tenfold(model, weights_list,
        data_folds, label_folds, id_folds, mode=mode)

    return param_dir.name, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # 1. eval用configの読み込み
    eval_cfg = load_config(args.config)
    experiment_dir = Path(eval_cfg.experiment_dir)
    output_dir = Path(eval_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "eval_config.lock.yaml")
    # 2. 訓練時のconfig.lock.yamlを読み込み
    train_cfg = load_config(experiment_dir / "config.lock.yaml")
    shutil.copy(experiment_dir / "config.lock.yaml", output_dir / "train_config.lock.yaml")
    modes = eval_cfg.mode  # リスト形式
    target_params = getattr(eval_cfg, 'target_params', None)

    # 3. foldループ
    for fold in train_cfg.folds:
        data_source = Path(train_cfg.data_source_base_dir) / fold
        fold_dir = experiment_dir / fold

        # データロード
        data_folds, label_folds, id_folds = tenfold_data_loader(data_source)
        param_dirs = list_param_dirs(fold_dir, target_params)

        # 4. modeループ
        for mode in modes:
            # paramループ単位で並列実行
            with ProcessPoolExecutor(max_workers=eval_cfg.workers) as executor:
                futures = []
                for param_dir in param_dirs:
                    future = executor.submit(eval_single_param, train_cfg, param_dir, data_folds, label_folds, id_folds, mode)
                    futures.append(future)

                for future in futures:
                    param_name, results = future.result()
                    # 結果を保存
                    result_path = output_dir / fold / param_name / f"{mode}_results.json"
                    save_eval_results(results, str(result_path))
                    print(f"fold={fold}, params={param_name}, mode={mode} completed")

    return


if __name__ == "__main__":
    main()