import numpy as np
import json
from pathlib import Path
from datetime import datetime


def save_fold_data(original_npz_path: Path, fold_samples: list, output_path: Path, fold_id: str):
    """指定されたfoldのサンプルを元のnpzから抽出して保存
    
    Args:
        original_npz_path: 元のnpzファイルのパス
        fold_samples: 保存するサンプルのリスト (divide_into_10foldsの出力)
        output_path: 出力先のnpzファイルパス
        fold_id: fold ID (例: "a_b")
    """
    # 元データを読み込み
    original_data = np.load(original_npz_path)
    
    # 新しいnpzファイルに保存するデータを準備
    save_dict = {}
    for new_idx, sample_info in enumerate(fold_samples):
        original_idx = sample_info['index']
        save_dict[f'sample_{new_idx}_data'] = original_data[f'sample_{original_idx}_data']
        save_dict[f'sample_{new_idx}_class'] = original_data[f'sample_{original_idx}_class']
        save_dict[f'sample_{new_idx}_id'] = original_data[f'sample_{original_idx}_id']
    
    # メタデータを追加
    save_dict['num_samples'] = len(fold_samples)
    save_dict['fold_id'] = fold_id
    
    # npzファイルとして保存
    np.savez(output_path, **save_dict)


def load_samples(npz_path: Path):
    """npzファイルからサンプル情報を取得してクラスでソート"""
    data = np.load(npz_path)
    sample_keys = [key for key in data.keys() if key.startswith('sample_') and key.endswith('_data')]
    
    samples = []
    for i in range(len(sample_keys)):
        samples.append({
            'index': i,
            'class': int(data[f'sample_{i}_class']),
            'id': str(data[f'sample_{i}_id']),
            'shape': data[f'sample_{i}_data'].shape
        })
    
    samples.sort(key=lambda x: (x['class'], x['id']))
    return samples


def divide_into_10folds(samples):
    """各クラスのデータを10グループに分割してクラスごとに統合"""
    # クラスごとに分類
    class_samples = {cls: [s for s in samples if s['class'] == cls] for cls in [0, 1, 2]}
    
    # 各クラスを10グループに分割 (9個×5 + 8個×5)
    folds_per_class = {}
    for cls in [0, 1, 2]:
        samples_in_class = class_samples[cls]
        folds_per_class[cls] = [samples_in_class[i:i+9] for i in range(0, 45, 9)] + \
                                [samples_in_class[i:i+8] for i in range(45, 85, 8)]
    
    # 分割結果を表示
    print("=== 各クラスの分割 ===")
    for cls in [0, 1, 2]:
        fold_sizes = [len(fold) for fold in folds_per_class[cls]]
        print(f"class {cls}: {fold_sizes}")
    
    # 3クラスを統合
    print("\n=== 統合後のfold構成 ===")
    combined_folds = []
    for fold_idx in range(10):
        fold_data = []
        for cls in [0, 1, 2]:
            fold_data.extend(folds_per_class[cls][fold_idx])
        combined_folds.append(fold_data)
        
        # 統計情報を表示
        class_counts = {cls: sum(1 for s in fold_data if s['class'] == cls) for cls in [0, 1, 2]}
        total = len(fold_data)
        print(f"fold {fold_idx}: total={total:2d} (class0={class_counts[0]}, class1={class_counts[1]}, class2={class_counts[2]})")
    
    print()
    return combined_folds


def create_metadata(source_fold_char: str, combined_folds: list, original_metadata: dict):
    """各foldのメタデータを生成
    
    Args:
        source_fold_char: 元のfold文字 (a~j)
        combined_folds: divide_into_10foldsの出力
        original_metadata: 元のmetadata.jsonの内容
    
    Returns:
        メタデータ辞書
    """
    # 元のfoldからサンプルID→クラスのマッピングを取得
    source_fold_key = f"fold_{source_fold_char}"
    original_samples = original_metadata["folds"][source_fold_key]
    
    # 各divisionのメタデータを構築
    divisions = {}
    division_stats = []
    fold_chars = 'abcdefghij'
    
    for fold_idx, fold_samples in enumerate(combined_folds):
        # 数字のインデックスを文字に変換
        fold_char_name = fold_chars[fold_idx]
        
        # このdivisionに含まれるサンプルのID→クラスマッピング
        fold_dict = {}
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for sample_info in fold_samples:
            sample_id = sample_info['id']
            sample_class = sample_info['class']
            fold_dict[sample_id] = sample_class
            class_counts[sample_class] += 1
        
        divisions[f"fold_{fold_char_name}"] = fold_dict
        
        # 統計情報
        division_stats.append({
            "division_id": fold_char_name,
            "num_samples": len(fold_samples),
            "class_distribution": {
                "0": class_counts[0],
                "1": class_counts[1],
                "2": class_counts[2]
            }
        })
    
    # メタデータ構造を作成
    metadata = {
        "created_at": datetime.now().isoformat(),
        "format": "npz_ragged",
        "description": f"10-fold division of fold_{source_fold_char} ESN training data",
        "source_fold": source_fold_char,
        "num_divisions": 10,
        "divisions": divisions,
        "division_stats": division_stats
    }
    
    return metadata


def main():
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "10fold_npy"
    output_base_dir = Path(__file__).parent.parent.parent / "dataset" / "10fold_npy_div"
    
    # 元のmetadata.jsonを読み込み
    metadata_file = dataset_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"警告: {metadata_file} が見つかりません。メタデータは生成されません。")
        original_metadata = None
    else:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            original_metadata = json.load(f)
    
    # 数字のインデックスを文字に変換するためのマッピング
    fold_chars = 'abcdefghij'
    
    for fold_char in fold_chars:
        file_name = f"fold_{fold_char}.npz"
        npz_file = dataset_dir / file_name
        
        print(f"\n{'='*80}")
        print(f"処理中: {file_name}")
        print(f"{'='*80}\n")
        
        if not npz_file.exists():
            print(f"エラー: {npz_file} が見つかりません")
            continue
        
        samples = load_samples(npz_file)
        combined_folds = divide_into_10folds(samples)
        
        # 出力ディレクトリを作成
        output_dir = output_base_dir / fold_char
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各foldをファイルとして保存
        print(f"保存中: {output_dir}")
        for fold_idx, fold_samples in enumerate(combined_folds):
            # 数字のインデックスを文字に変換 (0→a, 1→b, ..., 9→j)
            fold_char_name = fold_chars[fold_idx]
            output_file = output_dir / f"fold_{fold_char_name}.npz"
            # fold_idは "元のfold_新しいfold" の形式 (例: "a_b")
            fold_id = f"{fold_char}_{fold_char_name}"
            save_fold_data(npz_file, fold_samples, output_file, fold_id)
            print(f"  - fold_{fold_char_name}.npz 保存完了 (samples: {len(fold_samples)})")
        
        # メタデータを生成・保存
        if original_metadata:
            metadata = create_metadata(fold_char, combined_folds, original_metadata)
            metadata_output = output_dir / "metadata.json"
            with open(metadata_output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"  - metadata.json 保存完了")
        
        print()


if __name__ == "__main__":
    main()
