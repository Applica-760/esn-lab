from pathlib import Path
import numpy as np


def tenfold_data_loader(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    npz_files = sorted(dataset_dir.glob("*.npz"))
    
    # 全foldのデータを格納する3つのリスト
    data_folds = []
    label_folds = []
    id_folds = []
    
    # 各NPZファイルを処理
    for npz_file in npz_files:
        
        with np.load(npz_file, allow_pickle=True) as npz_data:
            num_samples = int(npz_data["num_samples"])
            
            # 各サンプルのデータをリストに格納
            data_list = []
            label_list = []
            id_list = []
            
            for i in range(num_samples):
                data_list.append(npz_data[f"{i}_data"])
                label_list.append(npz_data[f"{i}_label"])
                id_list.append(str(npz_data[f"{i}_id"]))
            
            # 各リストに追加
            data_folds.append(data_list)
            label_folds.append(label_list)
            id_folds.append(id_list)
    
    print("dataset loaded")
    
    return data_folds, label_folds, id_folds
