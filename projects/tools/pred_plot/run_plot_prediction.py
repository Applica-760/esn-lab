import json
from pathlib import Path

from projects.tools.pred_plot.prediction_cli import (
    run_interactive_cli, 
    build_output_path, 
    prompt_path,
    prompt_confirm,
)
from projects.utils.eval.filter import group_targets_by_source
from projects.utils.eval.plot_prediction import plot_prediction


"""
python -m projects.tools.pred_plot.run_plot_prediction
"""

# デフォルトのpred_results.json格納ディレクトリ
DEFAULT_PRED_RESULT_DIR = Path("outputs/experiments/pred_results")
# デフォルトの評価結果格納ディレクトリ（judgment_results_{mode}.csvがある階層）
DEFAULT_EVAL_DIR = Path("outputs/analysis/confusion_matrix")


def select_data_sources() -> tuple:
    """
    データソースの選択
    """
    print("=" * 50)
    print("  Prediction Plot Tool")
    print("=" * 50)
    
    print("\n=== [Step 0] データソース設定 ===")
    
    # pred_result_dir
    pred_result_dir = prompt_path(
        "\npred_results.json の格納ディレクトリ:",
        DEFAULT_PRED_RESULT_DIR
    )
    
    if not pred_result_dir.exists():
        print(f"警告: {pred_result_dir} が存在しません")
        if not prompt_confirm("続行しますか?", default=False):
            return None, None
    
    # eval_dir (judgment_results_{mode}.csv の格納ディレクトリ)
    eval_dir = prompt_path(
        "\njudgment_results_{mode}.csv の格納ディレクトリ (パラメータディレクトリがある階層):",
        DEFAULT_EVAL_DIR
    )
    
    if not eval_dir.exists():
        print(f"エラー: {eval_dir} が存在しません")
        return None, None
    
    return pred_result_dir, eval_dir


# =============================================================================
# プロット実行
# =============================================================================

def execute_plots(config: dict) -> None:
    """
    CLIから取得した設定に基づいてプロットを実行
    """
    param_name = config["param_name"]
    mode = config["mode"]
    targets = config["targets"]
    output_dir = config["output_dir"]
    pred_result_dir = config["pred_result_dir"]
    
    # (group, fold_index) でグルーピング
    grouped = group_targets_by_source(targets)
    
    total = len(targets)
    done = 0
    
    print(f"\n=== プロット実行中 ===")
    
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
                output_path = build_output_path(
                    output_dir, param_name, group, fold_index, sample_id
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    plot_prediction(results, sample_id, str(output_path), ext="png")
                    print(f"[{done}/{total}] {sample_id} ... done")
                except ValueError as e:
                    print(f"[{done}/{total}] {sample_id} ... エラー: {e}")
    
    print(f"\n完了: {total}件のプロットを保存しました")
    print(f"出力先: {output_dir}")


# =============================================================================
# メイン
# =============================================================================

def main():
    """
    メイン関数
    
    1. データソース選択
    2. 対話型CLIでフィルタ条件を収集
    3. プロット実行
    """
    # Step 0: データソース選択
    pred_result_dir, eval_dir = select_data_sources()
    
    if pred_result_dir is None:
        print("キャンセルしました")
        return
    
    # 対話型CLI実行
    config = run_interactive_cli(pred_result_dir, eval_dir)
    
    if config is None:
        return
    
    # プロット実行
    execute_plots(config)


if __name__ == "__main__":
    main()