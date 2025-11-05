"""

python tools/data_analysis/analyze_behavior_shift.py \
    /home/takumi/share/esn-lab/dataset/complements-binary \
    --csv data/get_300seqs.csv

"""
import argparse
from pathlib import Path
import pandas as pd


# 行動が途中で切り替わるデータのリストを取得
def get_non_uniform_npy_files(directory, csv_path):
    dir_path = Path(directory)
    csv_path = Path(csv_path)
    
    df = pd.read_csv(csv_path)
    
    csv_dict = {}
    for _, row in df.iterrows():
        basename = Path(row['file_path']).stem
        csv_dict[basename] = row
    
    npy_files = sorted(dir_path.glob("*.npy"))
    
    non_uniform_files = []
    
    for npy_file in npy_files:
        basename = npy_file.stem
        
        if basename in csv_dict:
            row = csv_dict[basename]
            uniform_flag = row['uniform_flag']
            
            if uniform_flag != 1:
                non_uniform_files.append(basename)
    
    return non_uniform_files


def main():
    parser = argparse.ArgumentParser(description=".npyファイルを基準として、uniform_flagが1でないファイル名のリストを取得する")
    parser.add_argument("directory", type=str, help=".npyファイルが格納されているディレクトリパス")
    parser.add_argument("--csv",type=str,required=True,help="file_path列とuniform_flag列を含むCSVファイルのパス")
    args = parser.parse_args()
    
    try:
        non_uniform_files = get_non_uniform_npy_files(args.directory, args.csv)
        print(non_uniform_files)
    except Exception as e:
        print(f"エラー: {e}")
        return


if __name__ == "__main__":
    main()
