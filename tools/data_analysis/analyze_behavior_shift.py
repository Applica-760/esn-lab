"""
行動が途中で切り替わるデータの遷移パターンを分析・可視化するスクリプト

使用例:
python tools/data_analysis/analyze_behavior_shift.py \
    /home/takumi/share/esn-lab/dataset/complements-binary \
    --csv data/get_300seqs.csv \
    --output-dir data/transition_analysis2

出力:
    - filtered_series.csv: フィルタリング後の系列データ
    - transition_patterns.csv: 遷移パターンの要約
    - transition_durations.csv: 各遷移の持続時間統計
    - sankey_diagram.png: サンキーダイアグラム
    - transition_heatmap.png: 遷移頻度のヒートマップ
"""
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib  # 日本語フォント対応

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 状態ラベルの定義
STATE_LABELS = {
    '0': 'other',
    '1': 'foraging',
    '2': 'rumination'
}

# 状態ごとの色設定
STATE_COLORS = {
    'other': '#4A90E2',        # 青系
    'foraging': '#50C878',      # 緑系
    'rumination': '#FF8C42'     # オレンジ系
}

# 始点ラベル別の薄い色設定
START_COLORS = {
    'other': '#D6E9F8',      # 薄い青
    'foraging': '#D4EDDA',   # 薄い緑
    'rumination': '#FFE5D0'  # 薄いオレンジ
}


def get_state_label(state):
    """状態番号からラベル名を取得"""
    return STATE_LABELS.get(str(state), str(state))


# ================================================================================
# データ読み込み・前処理関連
# ================================================================================

def load_csv_mapping(csv_path):
    """
    CSVファイルを読み込み、file_pathのベースネームをキーとした辞書を作成
    
    Args:
        csv_path: CSVファイルのパス
    
    Returns:
        dict: {basename: row} の辞書
    """
    df = pd.read_csv(csv_path)
    csv_dict = {}
    for _, row in df.iterrows():
        basename = Path(row['file_path']).stem
        csv_dict[basename] = row
    return csv_dict


def get_non_uniform_files(directory, csv_path):
    """
    行動が途中で切り替わるデータ(uniform_flag != 1)のファイル名リストを取得
    
    Args:
        directory: .npyファイルが格納されているディレクトリパス
        csv_path: file_path列とuniform_flag列を含むCSVファイルのパス
    
    Returns:
        list: uniform_flag != 1のファイルのベースネームリスト
    """
    csv_dict = load_csv_mapping(csv_path)
    non_uniform_files = []
    
    for basename, row in csv_dict.items():
        if row['uniform_flag'] != 1:
            non_uniform_files.append(basename)
    
    return non_uniform_files


def expand_300_to_9000(converted_300):
    """
    300桁の数字を9000桁に展開する
    各数字を30回繰り返す
    
    Args:
        converted_300: 300桁の文字列
    
    Returns:
        9000桁の数字のリスト
    """
    expanded = []
    for digit in converted_300:
        expanded.extend([int(digit)] * 30)
    return expanded


def process_and_filter_data(directory, csv_path, output_path):
    """
    .npyファイルとcsvデータを対応づけ、全成分が255の時刻を削除して出力
    
    Args:
        directory: .npyファイルが格納されているディレクトリパス
        csv_path: file_path列とconverted_300列を含むCSVファイルのパス
        output_path: 結果を出力するCSVファイルのパス
    """
    dir_path = Path(directory)
    
    # CSVを読み込んでconverted_300列を取得
    csv_dict = load_csv_mapping(csv_path)
    converted_dict = {basename: row['converted_300'] for basename, row in csv_dict.items()}
    
    # 対象ファイル名のリストを取得
    non_uniform_files = get_non_uniform_files(directory, csv_path)
    
    results = []
    
    # 各ファイルを処理
    for filename in non_uniform_files:
        npy_path = dir_path / f"{filename}.npy"
        
        if not npy_path.exists():
            print(f"警告: {npy_path} が見つかりません")
            continue
        
        if filename not in converted_dict:
            print(f"警告: {filename} がCSVに見つかりません")
            continue
        
        # .npyファイルを読み込み、256×9000に切り詰め
        npy_data = np.load(npy_path)[:, :9000]
        
        # csvの300桁を9000桁に展開
        csv_series = expand_300_to_9000(converted_dict[filename])
        
        # フィルタリング: 256成分全てが255でない時刻のみを残す
        filtered_csv = [
            csv_series[t] for t in range(9000)
            if not np.all(npy_data[:, t] == 255)
        ]
        
        # 結果を記録
        results.append({
            'npy_file_path': str(npy_path),
            'filtered_series': ''.join(map(str, filtered_csv)),
            'series_length': len(filtered_csv)
        })
        
        print(f"処理完了: {filename} (元: 9000, フィルタ後: {len(filtered_csv)})")
    
    # 結果をDataFrameに変換してCSV出力
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"\n結果を {output_path} に出力しました")



# ================================================================================
# 遷移パターン分析関連
# ================================================================================

def extract_transitions(series_str):
    """
    系列から状態遷移を抽出する
    
    Args:
        series_str: 状態系列の文字列 (例: "0000222211110")
    
    Returns:
        tuple: (transitions, pattern)
            - transitions: 遷移リスト [(from_state, to_state, duration), ...]
            - pattern: 遷移パターン文字列 (例: "other→rumination→foraging→other")
    """
    if not series_str:
        return [], ""
    
    transitions = []
    pattern_states = [series_str[0]]
    current_state = series_str[0]
    current_duration = 1
    
    for char in series_str[1:]:
        if char == current_state:
            current_duration += 1
        else:
            # 遷移が発生
            transitions.append((current_state, char, current_duration))
            pattern_states.append(char)
            current_state = char
            current_duration = 1
    
    # ラベル名を使ったパターン文字列を作成
    pattern = "→".join([get_state_label(s) for s in pattern_states])
    
    return transitions, pattern


def analyze_transitions(df):
    """
    全データの遷移パターンを分析
    
    Args:
        df: filtered_series列を持つDataFrame
    
    Returns:
        tuple: (pattern_summary, transition_stats, all_transitions)
            - pattern_summary: パターンごとの集計結果のDataFrame
            - transition_stats: 遷移ごとの統計情報のDataFrame
            - all_transitions: 全遷移の詳細データのDataFrame
    """
    pattern_counts = defaultdict(int)
    transition_durations = defaultdict(list)
    transition_counts = defaultdict(int)
    all_transitions = []
    
    # 全データの遷移を抽出
    for _, row in df.iterrows():
        transitions, pattern = extract_transitions(row['filtered_series'])
        pattern_counts[pattern] += 1
        
        # 各遷移の情報を記録
        for from_state, to_state, duration in transitions:
            transition_key = f"{get_state_label(from_state)}→{get_state_label(to_state)}"
            transition_durations[transition_key].append(duration)
            transition_counts[transition_key] += 1
            all_transitions.append({
                'file_path': row['npy_file_path'],
                'from_state': get_state_label(from_state),
                'to_state': get_state_label(to_state),
                'duration': duration,
                'pattern': pattern
            })
    
    # パターン要約の作成（頻度順にソート）
    pattern_summary = pd.DataFrame([
        {
            'pattern': pattern,
            'count': count,
            'percentage': count / len(df) * 100
        }
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    # 遷移ごとの統計の作成
    transition_stats = pd.DataFrame([
        {
            'transition': transition_key,
            'count': transition_counts[transition_key],
            'mean_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'std_duration': np.std(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'q25_duration': np.percentile(durations, 25),
            'q75_duration': np.percentile(durations, 75)
        }
        for transition_key, durations in sorted(transition_durations.items())
    ])
    
    return pattern_summary, transition_stats, pd.DataFrame(all_transitions)



# ================================================================================
# 遷移行列作成関連
# ================================================================================

def build_transition_matrix(transition_stats_df):
    """
    遷移統計DataFrameから遷移行列を作成
    
    Args:
        transition_stats_df: 遷移統計のDataFrame
    
    Returns:
        tuple: (transition_matrix, duration_matrix, state_labels)
            - transition_matrix: 遷移回数の行列 (3x3)
            - duration_matrix: 平均持続時間の行列 (3x3)
            - state_labels: 状態ラベルのリスト
    """
    state_keys = ['1', '2', '0']  # foraging, rumination, other の順序
    state_labels = [get_state_label(s) for s in state_keys]
    transition_matrix = np.zeros((3, 3))
    duration_matrix = np.zeros((3, 3))
    
    for _, row in transition_stats_df.iterrows():
        from_label, to_label = row['transition'].split('→')
        from_idx = state_labels.index(from_label)
        to_idx = state_labels.index(to_label)
        transition_matrix[from_idx, to_idx] = row['count']
        duration_matrix[from_idx, to_idx] = row['mean_duration']
    
    return transition_matrix, duration_matrix, state_labels


# ================================================================================
# 可視化関連
# ================================================================================

def create_sankey_diagram(transition_stats_df, output_path):
    """
    サンキーダイアグラムを作成
    
    Args:
        transition_stats_df: 遷移統計のDataFrame
        output_path: 出力ファイルパス
    """
    transition_matrix, _, state_labels = build_transition_matrix(transition_stats_df)
    colors = [STATE_COLORS[label] for label in state_labels]
    
    # 可視化
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 各状態の位置を設定
    left_x, right_x = 0.25, 0.75
    y_positions = np.linspace(0.15, 0.85, len(state_labels))
    
    # 左側のラベル（起点: From）
    ax.text(left_x - 0.15, 0.95, 'From (Source)', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
    
    # 右側のラベル（遷移先: To）
    ax.text(right_x + 0.15, 0.95, 'To (Target)', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
    
    # 状態ノードを描画
    for i, label in enumerate(state_labels):
        # 左側（from）
        ax.text(left_x - 0.05, y_positions[i], label, 
                ha='right', va='center', fontsize=13, fontweight='bold',
                color=colors[i])
        ax.plot(left_x, y_positions[i], 'o', markersize=28, color=colors[i], 
                zorder=10, markeredgecolor='white', markeredgewidth=2)
        
        # 右側（to）
        ax.text(right_x + 0.05, y_positions[i], label, 
                ha='left', va='center', fontsize=13, fontweight='bold',
                color=colors[i])
        ax.plot(right_x, y_positions[i], 'o', markersize=28, color=colors[i], 
                zorder=10, markeredgecolor='white', markeredgewidth=2)
    
    # 遷移を描画
    max_count = transition_matrix.max()
    for i in range(len(state_labels)):
        for j in range(len(state_labels)):
            count = transition_matrix[i, j]
            if count > 0:
                # 線の太さを遷移頻度に比例させる
                linewidth = (count / max_count) * 15 + 2
                alpha = 0.35 + (count / max_count) * 0.45
                
                # 曲線を描画（起点の色を使用）
                ax.annotate('', xy=(right_x, y_positions[j]), 
                           xytext=(left_x, y_positions[i]),
                           arrowprops=dict(arrowstyle='->', lw=linewidth, 
                                         color=colors[i], alpha=alpha,
                                         connectionstyle="arc3,rad=0.3"),
                           zorder=5)
                
                # カウントを表示
                mid_x = (left_x + right_x) / 2
                mid_y = (y_positions[i] + y_positions[j]) / 2
                ax.text(mid_x, mid_y, f'{int(count)}', 
                       fontsize=11, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='gray', alpha=0.95, linewidth=1.5),
                       zorder=15, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('State Transition Sankey Diagram', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"サンキーダイアグラムを保存: {output_path}")


def style_table_cell(table, row, col, is_header=False, is_even_row=False, 
                     center_align=False, bold=False, gray_text=False):
    """
    テーブルのセルにスタイルを適用
    
    Args:
        table: matplotlibのテーブルオブジェクト
        row: 行番号
        col: 列番号
        is_header: ヘッダー行かどうか
        is_even_row: 偶数行かどうか
        center_align: 中央揃えにするか
        bold: 太字にするか
        gray_text: グレーテキストにするか
    """
    cell = table[(row, col)]
    
    if is_header:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    else:
        cell.set_facecolor('#E7E6E6' if is_even_row else '#F2F2F2')
        if gray_text:
            cell.set_text_props(color='#AAAAAA', weight='normal', fontsize=10)
        elif bold:
            cell.set_text_props(weight='bold')
    
    if center_align:
        cell.set_text_props(ha='center')
    
    cell.set_edgecolor('white')


def create_pattern_table(pattern_summary_df, output_path):
    """
    遷移パターンの上位10件をテーブル画像として作成
    
    Args:
        pattern_summary_df: パターン要約のDataFrame
        output_path: 出力ファイルパス
    """
    # 上位10件を取得・フォーマット
    top_patterns = pattern_summary_df.head(10).copy()
    top_patterns['percentage'] = top_patterns['percentage'].apply(lambda x: f'{x:.1f}%')
    top_patterns.columns = ['Transition Pattern', 'Count', 'Percentage']
    
    # テーブルデータの準備
    table_data = [['Rank', 'Transition Pattern', 'Count', 'Percentage']]
    for idx, (_, row) in enumerate(top_patterns.iterrows(), 1):
        table_data.append([str(idx), row['Transition Pattern'], 
                          str(row['Count']), row['Percentage']])
    
    # テーブル画像を作成
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.06, 0.228, 0.106, 0.126])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # スタイルを適用
    for i in range(len(table_data)):
        for j in range(4):
            style_table_cell(table, i, j, 
                           is_header=(i == 0),
                           is_even_row=(i % 2 == 0),
                           center_align=(j in [0, 2, 3]),
                           bold=(j in [0, 2] and i > 0))
    
    plt.title('Top 10 Transition Patterns', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"遷移パターンテーブルを保存: {output_path}")


def create_length_ranges():
    """
    データ長さ範囲のリストを生成
    
    Returns:
        list: [(lower, upper), ...] の範囲リスト
    """
    ranges = []
    for i in range(9000, 5000, -500):
        ranges.append((i - 499, i))
    ranges.append((1, 5000))  # 5000以下を1列にまとめる
    return ranges


def categorize_pattern(pattern):
    """
    パターンを分類（3回以上切り替わる場合はまとめる）
    
    Args:
        pattern: 遷移パターン文字列
    
    Returns:
        str: 分類後のパターン名
    """
    shift_count = pattern.count('→')
    return "More than 3 shifts" if shift_count >= 3 else pattern


def check_min_label_duration(series_str, min_duration):
    """
    系列内の各ラベルの連続時系列長が最低min_duration以上かチェック
    
    Args:
        series_str: 状態系列の文字列
        min_duration: 各ラベルの最低持続時間
    
    Returns:
        bool: 全てのラベルがmin_duration以上ならTrue
    """
    if not series_str:
        return False
    
    current_state = series_str[0]
    current_duration = 1
    
    for char in series_str[1:]:
        if char == current_state:
            current_duration += 1
        else:
            # 遷移が発生 - 前の状態の持続時間をチェック
            if current_duration < min_duration:
                return False
            current_state = char
            current_duration = 1
    
    # 最後のラベルの持続時間もチェック
    if current_duration < min_duration:
        return False
    
    return True


def create_pattern_by_length_table(filtered_df, output_path, min_label_duration=None):
    """
    行動切り替わりパターン × データ総長さ範囲のクロス集計テーブルを作成
    
    Args:
        filtered_df: filtered_series列とseries_length列を持つDataFrame
        output_path: 出力ファイルパス
        min_label_duration: 各ラベルの最低持続時間（Noneの場合は条件なし）
    """
    # 各データのパターンを抽出
    data_with_patterns = []
    for _, row in filtered_df.iterrows():
        # 最低持続時間の条件チェック
        if min_label_duration is not None:
            if not check_min_label_duration(row['filtered_series'], min_label_duration):
                continue  # 条件を満たさないデータはスキップ
        
        _, pattern = extract_transitions(row['filtered_series'])
        data_with_patterns.append({
            'pattern': categorize_pattern(pattern),
            'length': row['series_length']
        })
    
    pattern_df = pd.DataFrame(data_with_patterns)
    length_ranges = create_length_ranges()
    
    # 実際に存在するパターンをソート（"More than 3 shifts"は最後）
    unique_patterns = pattern_df['pattern'].unique().tolist()
    patterns_sorted = sorted([p for p in unique_patterns if p != "More than 3 shifts"])
    if "More than 3 shifts" in unique_patterns:
        patterns_sorted.append("More than 3 shifts")
    
    # クロス集計表を作成
    cross_table = []
    for pattern in patterns_sorted:
        row_data = {'Pattern': pattern}
        for lower, upper in length_ranges:
            # 該当するデータをカウント
            count = len(pattern_df[
                (pattern_df['pattern'] == pattern) & 
                (pattern_df['length'] >= lower) & 
                (pattern_df['length'] <= upper)
            ])
            if upper == 5000:
                col_name = f'≤{upper}'
            else:
                col_name = f'{upper}-{lower}'
            row_data[col_name] = count
        cross_table.append(row_data)
    
    cross_table_df = pd.DataFrame(cross_table)
    
    # テーブルデータの準備
    columns = ['Pattern']
    for lower, upper in length_ranges:
        if upper == 5000:
            columns.append(f'≤{upper}')
        else:
            columns.append(f'{upper}-{lower}')
    
    table_data = [columns]
    
    for _, row in cross_table_df.iterrows():
        row_list = [row['Pattern']]
        for col in columns[1:]:
            row_list.append(str(int(row[col])))
        table_data.append(row_list)
    
    # テーブル画像を作成
    fig, ax = plt.subplots(figsize=(20, max(6, len(patterns_sorted) * 0.35)))
    ax.axis('tight')
    ax.axis('off')
    
    # 列幅の設定
    pattern_col_width = 0.25 * (4/6)  # 6分の5に縮小
    length_col_width = (0.75 / len(length_ranges)) * (3/4)  # 4分の3に縮小
    col_widths = [pattern_col_width] + [length_col_width] * len(length_ranges)
    
    # テーブルの作成
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=col_widths)
    
    # スタイルの設定 - フォントサイズを大きく
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    
    # ヘッダー行のスタイル
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('white')
    
    # データ行のスタイル - 始点ラベル別に色分け
    for i in range(1, len(table_data)):
        pattern = table_data[i][0]
        
        # 行の背景色を決定
        if pattern == "More than 3 shifts":
            row_color = '#F0F0F0'  # グレー
        else:
            start_label = pattern.split('→')[0]
            row_color = START_COLORS.get(start_label, '#F2F2F2')
        
        for j in range(len(columns)):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('white')
            
            # パターン列は左揃え
            if j == 0:
                cell.set_text_props(ha='left', fontsize=11, weight='bold')
            else:
                # カウント列は中央揃え＋太字
                cell.set_text_props(ha='center', weight='bold', fontsize=11)
                # 値が0の場合は薄いグレー
                if table_data[i][j] == '0':
                    cell.set_text_props(color='#AAAAAA', weight='normal', fontsize=10)
    
    # タイトルに条件を追加
    title = 'Behavior Transition Pattern × Data Length Range'
    if min_label_duration is not None:
        title += f'\n(Min label duration: {min_label_duration} timesteps)'
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 出力メッセージ
    message = f"パターン×長さ範囲テーブルを保存: {output_path}"
    if min_label_duration is not None:
        message += f" (最低ラベル持続時間: {min_label_duration}, データ数: {len(pattern_df)})"
    else:
        message += f" (条件なし, データ数: {len(pattern_df)})"
    print(message)


def add_heatmap_annotations(ax, matrix, state_labels, is_duration=False):
    """
    ヒートマップに値のアノテーションを追加
    
    Args:
        ax: matplotlibのaxesオブジェクト
        matrix: 表示する行列
        state_labels: 状態ラベルのリスト
        is_duration: 持続時間のマトリクスかどうか
    """
    if is_duration:
        matrix_masked = matrix.copy()
        matrix_masked[np.isnan(matrix_masked)] = -1  # NaNを-1に変換して判定
    
    for i in range(len(state_labels)):
        for j in range(len(state_labels)):
            if is_duration:
                if matrix_masked[i, j] >= 0:
                    duration = matrix[i, j]
                    max_val = np.nanmax(matrix)
                    min_val = np.nanmin(matrix)
                    normalized = (duration - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    text_color = 'white' if normalized > 0.5 else 'black'
                    ax.text(j, i, f'{duration:.0f}', ha='center', va='center',
                           color=text_color, fontsize=13, fontweight='bold')
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                           color='#666666', fontsize=10, style='italic')
            else:
                count = int(matrix[i, j])
                if count > 0:
                    max_val = matrix.max()
                    text_color = 'white' if matrix[i, j] > max_val * 0.5 else 'black'
                    ax.text(j, i, f'{count}', ha='center', va='center',
                           color=text_color, fontsize=13, fontweight='bold')
                else:
                    ax.text(j, i, '0', ha='center', va='center',
                           color='#666666', fontsize=11)


def create_transition_heatmap(transition_stats_df, output_path):
    """
    遷移頻度のヒートマップを作成（青色の濃淡で統一）
    
    Args:
        transition_stats_df: 遷移統計のDataFrame
        output_path: 出力ファイルパス
    """
    transition_matrix, duration_matrix, state_labels = build_transition_matrix(transition_stats_df)
    
    # 2つのサブプロットを作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 遷移頻度のヒートマップ
    im1 = ax1.imshow(transition_matrix, cmap='Blues', aspect='auto', vmin=0)
    ax1.set_xticks(range(len(state_labels)))
    ax1.set_yticks(range(len(state_labels)))
    ax1.set_xticklabels([f'To: {s}' for s in state_labels], fontsize=11)
    ax1.set_yticklabels([f'From: {s}' for s in state_labels], fontsize=11)
    ax1.set_title('Transition Frequency', fontsize=14, fontweight='bold', pad=15)
    add_heatmap_annotations(ax1, transition_matrix, state_labels, is_duration=False)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Number of Transitions')
    cbar1.ax.tick_params(labelsize=10)
    
    # 平均持続時間のヒートマップ
    duration_matrix_masked = duration_matrix.copy()
    duration_matrix_masked[transition_matrix == 0] = np.nan
    im2 = ax2.imshow(duration_matrix_masked, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(state_labels)))
    ax2.set_yticks(range(len(state_labels)))
    ax2.set_xticklabels([f'To: {s}' for s in state_labels], fontsize=11)
    ax2.set_yticklabels([f'From: {s}' for s in state_labels], fontsize=11)
    ax2.set_title('Mean Duration Before Transition', fontsize=14, fontweight='bold', pad=15)
    add_heatmap_annotations(ax2, duration_matrix_masked, state_labels, is_duration=True)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Mean Duration (timesteps)')
    cbar2.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ヒートマップを保存: {output_path}")



# ================================================================================
# メイン処理関連
# ================================================================================

def save_analysis_results(pattern_summary, transition_stats, all_transitions, output_dir):
    """
    分析結果をCSVファイルとして保存
    
    Args:
        pattern_summary: パターン要約のDataFrame
        transition_stats: 遷移統計のDataFrame
        all_transitions: 全遷移の詳細データのDataFrame
        output_dir: 出力ディレクトリ
    """
    pattern_path = output_dir / "transition_patterns.csv"
    pattern_summary.to_csv(pattern_path, index=False)
    print(f"\n遷移パターン要約を保存: {pattern_path}")
    print(f"  総パターン数: {len(pattern_summary)}")
    print(f"  上位5パターン:")
    for _, row in pattern_summary.head(5).iterrows():
        print(f"    {row['pattern']}: {row['count']}回 ({row['percentage']:.1f}%)")
    
    duration_path = output_dir / "transition_durations.csv"
    transition_stats.to_csv(duration_path, index=False)
    print(f"\n遷移持続時間統計を保存: {duration_path}")
    
    details_path = output_dir / "transition_details.csv"
    all_transitions.to_csv(details_path, index=False)
    print(f"遷移詳細データを保存: {details_path}")


def create_visualizations(filtered_df, output_dir):
    """
    全ての可視化を実行
    
    Args:
        filtered_df: filtered_series列を持つDataFrame
        output_dir: 出力ディレクトリ
    """
    print("\n遷移パターンを分析中...")
    pattern_summary, transition_stats, all_transitions = analyze_transitions(filtered_df)
    
    # CSVとして保存
    save_analysis_results(pattern_summary, transition_stats, all_transitions, output_dir)
    
    # 可視化
    print("\n可視化を作成中...")
    create_pattern_table(pattern_summary, output_dir / "transition_patterns_table.png")
    create_sankey_diagram(transition_stats, output_dir / "sankey_diagram.png")
    create_transition_heatmap(transition_stats, output_dir / "transition_heatmap.png")
    
    # パターン×長さ範囲のテーブルを複数条件で作成
    print("\nパターン×長さ範囲のテーブルを作成中...")
    create_pattern_by_length_table(filtered_df, output_dir / "pattern_by_length_table.png")
    create_pattern_by_length_table(filtered_df, output_dir / "pattern_by_length_table_min1000.png", min_label_duration=1000)
    create_pattern_by_length_table(filtered_df, output_dir / "pattern_by_length_table_min3000.png", min_label_duration=3000)


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="行動が途中で切り替わるデータの遷移パターンを分析・可視化する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
出力ファイル:
  filtered_series.csv                  - フィルタリング後の系列データ
  transition_patterns.csv              - 遷移パターンの要約（頻度、割合）
  transition_patterns_table.png        - 遷移パターンTop10のテーブル画像
  transition_durations.csv             - 各遷移の持続時間統計
  transition_details.csv               - 全遷移の詳細データ
  sankey_diagram.png                   - 状態遷移のサンキーダイアグラム
  transition_heatmap.png               - 遷移頻度と平均持続時間のヒートマップ
  pattern_by_length_table.png          - 行動切り替わりパターン×データ総長さ範囲（条件なし）
  pattern_by_length_table_min1000.png  - 同上（各ラベル最低1000timesteps以上）
  pattern_by_length_table_min3000.png  - 同上（各ラベル最低3000timesteps以上）
        """
    )
    parser.add_argument("directory", type=str, 
                       help=".npyファイルが格納されているディレクトリパス")
    parser.add_argument("--csv", type=str, required=True, 
                       help="file_path列とconverted_300列を含むCSVファイルのパス")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="結果を出力するディレクトリパス")
    return parser.parse_args()


def main():
    """メインエントリーポイント"""
    args = parse_arguments()
    
    try:
        # 出力ディレクトリの作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力ディレクトリ: {output_dir}")
        
        # フィルタリング処理を実行
        print("\nデータのフィルタリング中...")
        filtered_csv_path = output_dir / "filtered_series.csv"
        process_and_filter_data(args.directory, args.csv, filtered_csv_path)
        
        # フィルタリング結果を読み込み
        filtered_df = pd.read_csv(filtered_csv_path)
        
        # 可視化と分析を実行
        create_visualizations(filtered_df, output_dir)
        
        print(f"\n完了！全ての結果は {output_dir} に保存されました。")
        return 0
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
