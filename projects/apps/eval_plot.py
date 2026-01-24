import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path

from projects.utils.app_init import setup_app_environment
from projects.utils.eval.judgment import load_judgment_results
from projects.utils.eval.plot_prediction import plot_prediction
from projects.utils.eval.filter import (
    apply_filters,
    apply_sampling,
    extract_ids_with_metadata,
    group_targets_by_source,
)
from projects.utils.weights import build_param_str

"""
python -m projects.apps.eval_plot --config projects/configs/eval_plot.yaml
"""


def prepare_plot_targets(cfg, param_name):
    """
    プロット対象の準備
    """
    # 必須項目のチェック
    required = ['pred_result_dir', 'eval_dir', 'output_dir', 'mode']
    for key in required:
        if not hasattr(cfg, key):
            raise ValueError(f"設定ファイルに '{key}' が定義されていません")
    
    # パスの存在確認
    pred_result_dir = Path(cfg.pred_result_dir)
    eval_dir = Path(cfg.eval_dir)
    
    if not pred_result_dir.exists():
        print(f"警告: pred_result_dir が存在しません: {pred_result_dir}")
    
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval_dir が存在しません: {eval_dir}")
    
    mode = cfg.mode
    
    # judgment_results ファイルの存在確認
    judgment_csv_path = eval_dir / param_name / f"judgment_results_{mode}.csv"
    
    if not judgment_csv_path.exists():
        raise FileNotFoundError(f"judgment_results が見つかりません: {judgment_csv_path}")
    
    # モードの妥当性チェック
    if mode not in ['train', 'test']:
        raise ValueError(f"mode は 'train' または 'test' である必要があります: {mode}")
    
    # サンプリング設定のチェック
    sampling = cfg.sampling
    if sampling['method'] not in ['all', 'random', 'first']:
        raise ValueError(f"sampling.method が不正です: {sampling['method']}")
    
    if sampling['method'] in ['random', 'first'] and not sampling.get('n'):
        raise ValueError(f"sampling.method が '{sampling['method']}' の場合、n を指定してください")
    
    # judgment_results の読み込み
    judgment_results = load_judgment_results(str(judgment_csv_path))
    
    print(f"judgment_results loaded: {len(judgment_results)} records")
    
    # フィルタ条件の構築
    filter_config = {
        "true_label": cfg.filters.get('true_label'),
        "pred_label": cfg.filters.get('pred_label'),
        "is_correct": cfg.filters.get('is_correct'),
        "groups": cfg.filters.get('groups'),
        "fold_indices": cfg.filters.get('fold_indices'),
    }
    
    # フィルタ適用
    filtered_results = apply_filters(judgment_results, filter_config)
    print(f"After filtering: {len(filtered_results)} records")
    
    if len(filtered_results) == 0:
        print("警告: フィルタ条件に合致するデータがありません")
        return []
    
    # サンプリング適用
    sampled_results = apply_sampling(
        filtered_results,
        method=sampling['method'],
        n=sampling.get('n'),
        seed=sampling.get('seed')
    )
    print(f"After sampling: {len(sampled_results)} records")
    
    # メタデータ付きID抽出
    targets = extract_ids_with_metadata(sampled_results)
    
    return targets


def execute_prediction_plots(cfg, param_name, targets):
    """
    プロット実行メインループ
    """
    pred_result_dir = Path(cfg.pred_result_dir)
    output_dir = Path(cfg.output_dir)
    mode = cfg.mode
    
    grouped = group_targets_by_source(targets)
    
    total = len(targets)
    done = 0
    
    for group, folds in grouped.items():
        for fold_index, ids in folds.items():
            # JSONを一度だけロード
            json_path = pred_result_dir / group / param_name / f"{mode}_results.json"
            
            if not json_path.exists():
                print(f"警告: {json_path} が見つかりません。スキップします。")
                done += len(ids)
                continue
            
            with open(json_path, 'r') as f:
                pred_results = json.load(f)
            
            # fold_indexに対応するresultsを取得
            fold_data = None
            for fd in pred_results:
                if fd["fold_index"] == fold_index:
                    fold_data = fd
                    break
            
            if fold_data is None:
                print(f"警告: fold_index={fold_index} が見つかりません。スキップします。")
                done += len(ids)
                continue
            
            results = fold_data["results"]
            
            # 各IDについてプロット
            for sample_id in ids:
                done += 1
                
                # 出力パスを生成
                output_path = output_dir / param_name / group / f"fold_{fold_index}" / sample_id
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    plot_prediction(results, sample_id, str(output_path), ext="png")
                    print(f"[{done}/{total}] {sample_id} ... done")
                except ValueError as e:
                    print(f"[{done}/{total}] {sample_id} ... エラー: {e}")
    

def main():
    cfg, _ = setup_app_environment()
    
    params = {'Nx': cfg.Nx, 'input_scale': cfg.input_scale, 'density': cfg.density, 'rho': cfg.rho}
    param_name = build_param_str(params)
    
    targets = prepare_plot_targets(cfg, param_name)
    if not targets:
        print("プロット対象がありません。処理を終了します。")
        return
    
    execute_prediction_plots(cfg, param_name, targets)
    
    print("plot finished")


if __name__ == "__main__":
    main()