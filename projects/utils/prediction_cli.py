"""
Prediction Plot 用の対話型CLI

ユーザーとの対話を通じてフィルタ条件・サンプリング設定を収集し、
プロット実行に必要な情報を返す。

出力先ディレクトリ構造:
    outputs/analysis/prediction_plots/
    └── {ユーザー指定のディレクトリ名}/
        └── {param_name}/
            └── {group}/
                └── fold_{fold_index}/
                    └── {id}.png

フィルタリングロジックは utils/filter.py に分離されており、
このモジュールはCUI（ユーザーインタラクション）のみを担当する。
"""

import os
import glob
import readline
from pathlib import Path
from typing import Optional

from simple_term_menu import TerminalMenu

from projects.utils.results import load_judgment_results
from projects.utils.weights import list_param_dirs
from projects.utils.filter import apply_filters, apply_sampling, extract_ids_with_metadata


# =============================================================================
# 定数
# =============================================================================

OUTPUT_BASE_DIR = Path("outputs/analysis/prediction_plots")

CLASS_NAMES = {
    0: "other",
    1: "foraging",
    2: "rumination",
}


# =============================================================================
# パス補完設定
# =============================================================================

def setup_path_completer():
    """
    readlineのTab補完をパス補完に設定
    """
    def path_completer(text, state):
        # ~を展開
        if text.startswith("~"):
            text = os.path.expanduser(text)
        
        # globパターンでマッチするパスを取得
        if text:
            pattern = text + "*"
        else:
            pattern = "*"
        
        matches = glob.glob(pattern)
        # ディレクトリには/を付ける
        matches = [m + "/" if os.path.isdir(m) else m for m in matches]
        
        try:
            return matches[state]
        except IndexError:
            return None
    
    readline.set_completer(path_completer)
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")


def reset_completer():
    """
    readlineの補完をリセット
    """
    readline.set_completer(None)


# =============================================================================
# 入力ヘルパー関数
# =============================================================================

def menu_select_one(title: str, options: list, default: int = 0) -> int:
    """
    単一選択メニュー（上下キーで選択）
    
    Args:
        title: メニュータイトル
        options: 選択肢のリスト
        default: デフォルト選択のインデックス
    
    Returns:
        選択されたインデックス
    """
    print(f"\n{title}")
    menu = TerminalMenu(
        options,
        cursor_index=default,
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("bg_cyan", "fg_black"),
    )
    idx = menu.show()
    
    # Ctrl+Cなどでキャンセルされた場合はデフォルトを返す
    if idx is None:
        return default
    return idx


def menu_select_multiple(title: str, options: list) -> list:
    """
    複数選択メニュー（スペースで選択、Enterで確定）
    
    Args:
        title: メニュータイトル
        options: 選択肢のリスト
    
    Returns:
        選択されたインデックスのリスト
    """
    print(f"\n{title}")
    print("  (スペースで選択/解除、Enterで確定)")
    
    menu = TerminalMenu(
        options,
        multi_select=True,
        show_multi_select_hint=True,
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("bg_cyan", "fg_black"),
        multi_select_cursor_style=("fg_green", "bold"),
        multi_select_select_on_accept=False,
    )
    selected = menu.show()
    
    if selected is None:
        return []
    
    # 単一選択の場合はタプルではなくintが返る
    if isinstance(selected, int):
        return [selected]
    
    return list(selected)


def prompt_int(prompt: str, default: int) -> int:
    """
    整数入力プロンプト
    """
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if user_input == "":
            return default
        try:
            return int(user_input)
        except ValueError:
            print("  数値を入力してください")


def prompt_str(prompt: str, default: str = "") -> str:
    """
    文字列入力プロンプト
    """
    default_display = f" [{default}]" if default else ""
    user_input = input(f"{prompt}{default_display}: ").strip()
    return user_input if user_input else default


def prompt_path(prompt: str, default: Path) -> Path:
    """
    パス入力プロンプト（Tab補完対応）
    """
    setup_path_completer()
    try:
        print(f"{prompt}")
        print(f"  デフォルト: {default}")
        user_input = input("パスを入力 (Enterでデフォルト、Tabで補完): ").strip()
        
        if user_input == "":
            return default
        return Path(os.path.expanduser(user_input))
    finally:
        reset_completer()


def prompt_confirm(prompt: str, default: bool = True) -> bool:
    """
    Yes/No確認プロンプト
    """
    options = ["Yes", "No"]
    default_idx = 0 if default else 1
    idx = menu_select_one(prompt, options, default=default_idx)
    return idx == 0


# =============================================================================
# 各ステップのCUI関数
# =============================================================================

def select_data_source(analysis_dir: Path) -> tuple:
    """
    [Step 1] データソース選択
    
    - 利用可能なパラメータディレクトリを一覧表示
    - ユーザーにパラメータを選択させる
    
    Returns:
        (param_dir: Path, param_name: str)
    """
    param_dirs = list_param_dirs(str(analysis_dir))
    param_names = [d.name for d in param_dirs]
    
    print("\n=== [Step 1] パラメータ選択 ===")
    print(f"analysis_dir: {analysis_dir}")
    
    idx = menu_select_one("利用可能なパラメータ:", param_names, default=0)
    
    return param_dirs[idx], param_names[idx]


def select_mode() -> str:
    """
    [Step 2] mode選択（train / test）
    
    Returns:
        "train" or "test"
    """
    print("\n=== [Step 2] mode選択 ===")
    options = ["train", "test"]
    idx = menu_select_one("modeを選択:", options, default=1)
    return options[idx]


def select_filter_true_label() -> Optional[int]:
    """
    [Step 3-1] 正解ラベルフィルタ選択
    
    Returns:
        選択されたラベル（0, 1, 2）またはNone（フィルタなし）
    """
    options = ["フィルタしない", "other", "foraging", "rumination"]
    idx = menu_select_one("正解ラベルでフィルタ?", options, default=0)
    
    if idx == 0:
        return None
    return idx - 1  # 0: other, 1: foraging, 2: rumination


def select_filter_correctness() -> Optional[bool]:
    """
    [Step 3-2] 判定結果（正誤）フィルタ選択
    
    Returns:
        True（正解のみ）, False（不正解のみ）, None（フィルタなし）
    """
    options = ["フィルタしない", "正解のみ", "不正解のみ"]
    idx = menu_select_one("判定結果でフィルタ?", options, default=0)
    
    if idx == 0:
        return None
    elif idx == 1:
        return True
    else:
        return False


def select_filter_pred_label() -> Optional[int]:
    """
    [Step 3-3] 予測ラベルフィルタ選択（不正解選択時のみ表示）
    
    Returns:
        選択されたラベルまたはNone
    """
    options = ["フィルタしない", "otherと誤判定", "foragingと誤判定", "ruminationと誤判定"]
    idx = menu_select_one("予測ラベルでフィルタ?", options, default=0)
    
    if idx == 0:
        return None
    return idx - 1


def select_filter_groups(available_groups: list) -> Optional[list]:
    """
    [Step 3-4] groupフィルタ選択
    
    Args:
        available_groups: 利用可能なgroupのリスト
    
    Returns:
        選択されたgroupのリスト、またはNone（全group）
    """
    options = ["全group", "特定のgroupを選択"]
    idx = menu_select_one("groupでフィルタ?", options, default=0)
    
    if idx == 0:
        return None
    
    # 複数選択モードへ
    indices = menu_select_multiple("groupを選択（複数可）:", available_groups)
    if not indices:
        return None
    
    return [available_groups[i] for i in indices]


def select_filter_folds(available_folds: list) -> Optional[list]:
    """
    [Step 3-5] fold_indexフィルタ選択
    
    Args:
        available_folds: 利用可能なfold_indexのリスト
    
    Returns:
        選択されたfold_indexのリスト、またはNone（全fold）
    """
    fold_options = [f"fold {f}" for f in available_folds]
    options = ["全fold", "特定のfoldを選択"]
    idx = menu_select_one("foldでフィルタ?", options, default=0)
    
    if idx == 0:
        return None
    
    # 複数選択モードへ
    indices = menu_select_multiple("foldを選択（複数可）:", fold_options)
    if not indices:
        return None
    
    return [available_folds[i] for i in indices]


def select_sampling(total_count: int) -> tuple:
    """
    [Step 4] サンプリング方法選択
    
    Args:
        total_count: フィルタ後の総件数
    
    Returns:
        (method: str, n: Optional[int])
        method: "all", "random", "first"
        n: サンプリング件数（method="all"の場合はNone）
    """
    print(f"\n=== [Step 4] サンプリング ===")
    print(f"条件に合致: {total_count}件")
    
    options = ["全件プロット", "ランダムにN件", "先頭N件"]
    idx = menu_select_one("サンプリング方法:", options, default=1 if total_count > 10 else 0)
    
    if idx == 0:
        return "all", None
    
    default_n = min(10, total_count)
    n = prompt_int("件数を入力", default=default_n)
    n = min(n, total_count)  # 総件数を超えないように
    
    if idx == 1:
        return "random", n
    else:
        return "first", n


def input_output_dirname() -> str:
    """
    [Step 5] 出力ディレクトリ名の入力
    
    outputs/analysis/prediction_plots/ 直下に作成するディレクトリ名を
    ユーザーに入力させる。
    
    Returns:
        ディレクトリ名（例: "foraging_misclassified"）
    """
    print(f"\n=== [Step 5] 出力ディレクトリ名 ===")
    print(f"出力先: {OUTPUT_BASE_DIR}/{{ディレクトリ名}}/...")
    
    while True:
        dirname = prompt_str("ディレクトリ名を入力", default="analysis_result")
        if dirname:
            return dirname
        print("  ディレクトリ名を入力してください")


def show_confirmation(filter_config: dict, sampling_config: dict, 
                      total_count: int, sample_count: int,
                      output_dir: Path) -> bool:
    """
    [Step 6] 実行確認
    
    選択された条件を表示し、実行するかどうかを確認する。
    
    Returns:
        True=実行する, False=キャンセル
    """
    print("\n=== [Step 6] 確認 ===")
    print("フィルタ条件:")
    
    if filter_config.get("true_label") is not None:
        print(f"  - 正解ラベル: {CLASS_NAMES[filter_config['true_label']]}")
    if filter_config.get("is_correct") is not None:
        print(f"  - 判定結果: {'正解' if filter_config['is_correct'] else '不正解'}")
    if filter_config.get("pred_label") is not None:
        print(f"  - 予測ラベル: {CLASS_NAMES[filter_config['pred_label']]}")
    if filter_config.get("groups") is not None:
        print(f"  - groups: {filter_config['groups']}")
    if filter_config.get("fold_indices") is not None:
        print(f"  - folds: {filter_config['fold_indices']}")
    
    if not any(v is not None for v in filter_config.values()):
        print("  (フィルタなし)")
    
    print(f"\nサンプリング: {sampling_config['method']}", end="")
    if sampling_config.get("n"):
        print(f" ({sampling_config['n']}件)")
    else:
        print()
    
    print(f"対象: {total_count}件中 {sample_count}件")
    print(f"出力先: {output_dir}")
    
    return prompt_confirm("\n実行しますか?", default=True)


# =============================================================================
# メインフロー
# =============================================================================

def run_interactive_cli(eval_result_dir: Path, analysis_dir: Path) -> Optional[dict]:
    """
    対話型CLIのメインフロー
    
    全ステップを順に実行し、プロット実行に必要な情報を収集する。
    
    Args:
        eval_result_dir: eval_results.jsonが格納されているディレクトリ
        analysis_dir: judgment_results.csvが格納されているディレクトリ
    
    Returns:
        プロット実行に必要な情報の辞書、またはNone（キャンセル時）
        {
            "param_name": str,
            "mode": str,
            "targets": list[dict],  # [{"id": str, "group": str, "fold_index": int}, ...]
            "output_dir": Path,
            "eval_result_dir": Path,
        }
    """
    print("=" * 50)
    print("  Prediction Plot Tool")
    print("=" * 50)
    
    # Step 1: パラメータ選択
    param_dir, param_name = select_data_source(analysis_dir)
    
    # Step 2: mode選択
    mode = select_mode()
    
    # judgment_results_{mode}.csv をロード
    judgment_csv_path = analysis_dir / param_name / f"judgment_results_{mode}.csv"
    if not judgment_csv_path.exists():
        print(f"\nエラー: {judgment_csv_path} が見つかりません")
        return None
    
    judgment_results = load_judgment_results(str(judgment_csv_path))
    
    # 利用可能なgroup/foldを取得
    available_groups = sorted(set(r["group"] for r in judgment_results))
    available_folds = sorted(set(r["fold_index"] for r in judgment_results))
    
    # Step 3: フィルタ条件
    print("\n=== [Step 3] フィルタ条件 ===")
    
    filter_config = {}
    
    filter_config["true_label"] = select_filter_true_label()
    filter_config["is_correct"] = select_filter_correctness()
    
    # 不正解を選択した場合のみ予測ラベルフィルタを表示
    if filter_config["is_correct"] is False:
        filter_config["pred_label"] = select_filter_pred_label()
    else:
        filter_config["pred_label"] = None
    
    filter_config["groups"] = select_filter_groups(available_groups)
    filter_config["fold_indices"] = select_filter_folds(available_folds)
    
    # フィルタ適用
    filtered_results = apply_filters(judgment_results, filter_config)
    total_count = len(filtered_results)
    
    if total_count == 0:
        print("\n条件に合致するデータがありません")
        return None
    
    # Step 4: サンプリング
    sampling_method, sampling_n = select_sampling(total_count)
    sampling_config = {"method": sampling_method, "n": sampling_n}
    
    sampled_results = apply_sampling(filtered_results, sampling_method, sampling_n)
    sample_count = len(sampled_results)
    
    # Step 5: 出力ディレクトリ名
    dirname = input_output_dirname()
    output_dir = OUTPUT_BASE_DIR / dirname
    
    # Step 6: 確認
    if not show_confirmation(filter_config, sampling_config, total_count, sample_count, output_dir):
        print("キャンセルしました")
        return None
    
    # 結果を返す
    targets = extract_ids_with_metadata(sampled_results)
    
    return {
        "param_name": param_name,
        "mode": mode,
        "targets": targets,
        "output_dir": output_dir,
        "eval_result_dir": eval_result_dir,
    }


# =============================================================================
# 出力パス生成
# =============================================================================

def build_output_path(output_dir: Path, param_name: str, 
                      group: str, fold_index: int, sample_id: str) -> Path:
    """
    プロット出力パスを生成
    
    構造: {output_dir}/{param_name}/{group}/fold_{fold_index}/{sample_id}
    
    Note: 拡張子はplot_prediction側で付与される
    
    Returns:
        出力ファイルパス（拡張子なし）
    """
    return output_dir / param_name / group / f"fold_{fold_index}" / sample_id
