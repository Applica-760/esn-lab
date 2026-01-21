
"""
Prediction Plot の実行スクリプト（エントリーポイント）

対話型CLIを通じてフィルタ条件を収集し、
条件に合致するサンプルのpredictionプロットを生成する。

使用方法:
    python -m projects.tools.data_analysis.run_plot_prediction

出力先:
    outputs/analysis/prediction_plots/{ユーザー指定}/
        └── {param_name}/{group}/fold_{fold_index}/{id}.png
"""

from pathlib import Path

from projects.utils.prediction_cli import (
    run_interactive_cli, 
    build_output_path, 
    prompt_path,
    prompt_confirm,
)
from projects.utils.filter import group_targets_by_source
from projects.utils.results import load_eval_results
from projects.utils.plot_prediction import plot_prediction


# =============================================================================
# 設定
# =============================================================================

# デフォルトのeval_results.json格納ディレクトリ
DEFAULT_EVAL_RESULT_DIR = Path("outputs/experiments/eval_results")

# デフォルトのanalysis結果格納ディレクトリ
DEFAULT_ANALYSIS_DIR = Path("outputs/analysis/prediction_plots")


# =============================================================================
# データソース選択
# =============================================================================

def select_data_sources() -> tuple:
    """
    データソースの選択
    
    Returns:
        (eval_result_dir: Path, analysis_dir: Path)
    """
    print("=" * 50)
    print("  Prediction Plot Tool")
    print("=" * 50)
    
    print("\n=== [Step 0] データソース設定 ===")
    
    # eval_result_dir
    eval_result_dir = prompt_path(
        "\neval_results.json の格納ディレクトリ:",
        DEFAULT_EVAL_RESULT_DIR
    )
    
    if not eval_result_dir.exists():
        print(f"警告: {eval_result_dir} が存在しません")
        if not prompt_confirm("続行しますか?", default=False):
            return None, None
    
    # analysis_dir (judgment_results.csv の格納ディレクトリ)
    analysis_dir = prompt_path(
        "\njudgment_results.csv の格納ディレクトリ (パラメータディレクトリがある階層):",
        DEFAULT_ANALYSIS_DIR
    )
    
    if not analysis_dir.exists():
        print(f"エラー: {analysis_dir} が存在しません")
        return None, None
    
    return eval_result_dir, analysis_dir


# =============================================================================
# プロット実行
# =============================================================================

def execute_plots(config: dict) -> None:
    """
    CLIから取得した設定に基づいてプロットを実行
    
    (group, fold_index) でグルーピングし、同じJSONは一度だけロードする最適化を行う。
    
    Args:
        config: run_interactive_cli() の戻り値
            {
                "param_name": str,
                "mode": str,
                "targets": list[dict],
                "output_dir": Path,
                "eval_result_dir": Path,
            }
    """
    param_name = config["param_name"]
    mode = config["mode"]
    targets = config["targets"]
    output_dir = config["output_dir"]
    eval_result_dir = config["eval_result_dir"]
    
    # (group, fold_index) でグルーピング
    grouped = group_targets_by_source(targets)
    
    total = len(targets)
    done = 0
    
    print(f"\n=== プロット実行中 ===")
    
    for group, folds in grouped.items():
        for fold_index, ids in folds.items():
            # JSONを一度だけロード
            json_path = eval_result_dir / group / param_name / f"{mode}_results.json"
            
            if not json_path.exists():
                print(f"警告: {json_path} が見つかりません。スキップします。")
                done += len(ids)
                continue
            
            eval_results = load_eval_results(str(json_path))
            
            # fold_indexに対応するresultsを取得
            fold_data = None
            for fd in eval_results:
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
    eval_result_dir, analysis_dir = select_data_sources()
    
    if eval_result_dir is None:
        print("キャンセルしました")
        return
    
    # 対話型CLI実行
    config = run_interactive_cli(eval_result_dir, analysis_dir)
    
    if config is None:
        return
    
    # プロット実行
    execute_plots(config)


if __name__ == "__main__":
    main()