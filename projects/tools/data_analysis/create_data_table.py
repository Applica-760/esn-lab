#!/usr/bin/env python3
"""
データ長ごとにラベル別データ点数のテーブルを作成

Usage:
python projects/tools/data_analysis/create_data_table.py \
  --csv data/get_300seqs.csv \
  --output-dir outputs/analysis/data_table
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 日本語フォント設定
plt.rcParams["font.family"] = "DejaVu Sans"


def extract_uniform_label(seq: str) -> int:
    """uniform_flag==1のデータからラベルを抽出"""
    uniq = set(seq)
    if len(uniq) != 1 or next(iter(uniq)) not in {"0", "1", "2"}:
        raise ValueError(f"Invalid sequence: {seq}")
    return int(next(iter(uniq)))


def assign_width_bin(width: int) -> tuple:
    """データ長を500刻みの階級に分類
    
    Returns:
        tuple: (下限, 上限) 9000以上の場合は (9000, None)
    """
    if width >= 9000:
        return (9000, None)
    
    lower = (width // 500) * 500
    upper = lower + 500
    return (lower, upper)


def create_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """データ長ごとのラベル別度数分布表を作成"""
    # ラベルと階級を抽出
    df = df.copy()
    df["label"] = df["converted_300"].apply(extract_uniform_label)
    df["bin"] = df["image_width_px"].apply(assign_width_bin)
    df["bin_lower"] = df["bin"].apply(lambda x: x[0])
    df["bin_upper"] = df["bin"].apply(lambda x: x[1])
    
    # 全階級を明示的に定義（0-500, 500-1000, ..., 8500-9000, 9000-）
    all_bins = []
    for i in range(0, 9000, 500):
        all_bins.append((i, i + 500))
    all_bins.append((9000, None))  # 9000以上
    
    # ラベル名（0=その他、1=採食、2=反芻）
    label_map = {0: "その他", 1: "採食", 2: "反芻"}
    
    # 各階級・各ラベルの度数を集計
    results = []
    for bin_lower, bin_upper in all_bins:
        if bin_upper is None:
            mask = df["bin_lower"] == bin_lower
        else:
            mask = (df["bin_lower"] == bin_lower) & (df["bin_upper"] == bin_upper)
        
        subset = df[mask]
        row = {"階級下限": bin_lower, "階級上限": bin_upper}
        for label_id, label_name in label_map.items():
            row[label_name] = int((subset["label"] == label_id).sum())
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # カラムの順序を調整（採食、反芻、その他の順に）
    cols = ["階級下限", "階級上限", "採食", "反芻", "その他"]
    result_df = result_df[cols]
    
    return result_df


def plot_frequency_chart(freq_table: pd.DataFrame, output_path: Path, format: str) -> None:
    """度数分布のヒストグラムと累積度数の折れ線グラフを描画
    
    Args:
        freq_table: 度数分布表のDataFrame
        output_path: 出力PDFファイルパス
    """
    # 階級ラベルを作成（9000-が左端になるよう逆順）
    labels = []
    for _, row in freq_table.iterrows():
        lower = int(row["階級下限"])
        upper = row["階級上限"]
        if pd.isna(upper):
            labels.append(f"{lower}-")
        else:
            labels.append(f"{lower}-{int(upper)}")
    
    # 逆順にする（9000-が左端）
    labels = labels[::-1]
    
    # 各ラベルの度数を取得（逆順）
    eating = freq_table["採食"].values[::-1]
    ruminating = freq_table["反芻"].values[::-1]
    other = freq_table["その他"].values[::-1]
    
    # 合計度数と累積度数を計算
    total = eating + ruminating + other
    cumulative = total.cumsum()
    cumulative_ratio = cumulative / cumulative[-1]  # 0〜1にスケール
    
    x = range(len(labels))
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 目盛りのフォントサイズを設定
    ax1.tick_params(axis='y', labelsize=13)
    
    # 棒グラフ（度数ヒストグラム）- 合計度数
    ax1.bar(x, total, color="#3758A5FF", label="Frequency")
    
    ax1.set_xlabel("Data Length (px)", fontsize=15)
    ax1.set_ylabel("Frequency", fontsize=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    
    # 累積度数の折れ線グラフ（第2軸、0〜1にスケール）
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_ratio, color="#D62525", marker="o", linewidth=2, 
             label="Cumulative Ratio")
    ax2.set_ylabel("Cumulative Ratio", fontsize=15)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelsize=13)
    
    # 両軸の凡例を1つにまとめてグラフ内上部中央に配置
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=2, fontsize=15)
    
    plt.tight_layout()
    
    # PDF保存
    fig.savefig(output_path, format=format, bbox_inches="tight")
    plt.close(fig)
    print(f"グラフを保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="データ長ごとにラベル別データ点数のテーブルを作成"
    )
    parser.add_argument("--csv", required=True, help="入力CSVファイルパス")
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリパス")
    args = parser.parse_args()
    
    # データ読み込み
    df = pd.read_csv(args.csv, dtype={"converted_300": str})
    
    # uniform_flag == 1 でフィルタリング
    df_filtered = df[df["uniform_flag"] == 1].reset_index(drop=True)
    
    # 度数分布表を作成
    freq_table = create_frequency_table(df_filtered)
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV保存
    output_csv_path = output_dir / "data_table.csv"
    freq_table.to_csv(output_csv_path, index=False)
    print(f"CSVを保存しました: {output_csv_path}")
    
    # グラフ描画・保存
    output_pdf_path = output_dir / "data_graph.pdf"
    plot_frequency_chart(freq_table, output_pdf_path, "pdf")
    output_pdf_path = output_dir / "data_graph.png"
    plot_frequency_chart(freq_table, output_pdf_path, "png")
    
    # コンソールにも表示
    print("\n=== 度数分布表 ===")
    print(freq_table.to_string(index=False))


if __name__ == "__main__":
    main()
