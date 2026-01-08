import os
import json
import numpy as np


def save_tenfold_weights(params: dict, weights_list: list, output_dir: str) -> None:
    # パラメータごとのサブディレクトリを作成
    param_str = f"Nx{params['Nx']}_dens{params['density']}_inscl{params['input_scale']}_rho{params['rho']}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)
    
    # パラメータをmetadata.jsonとして保存
    metadata_path = os.path.join(param_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    # 各foldの重みを個別に保存
    for i, weight in enumerate(weights_list):
        filename = f"fold{i}.npz"
        filepath = os.path.join(param_dir, filename)
        np.savez(filepath, weight=weight)
