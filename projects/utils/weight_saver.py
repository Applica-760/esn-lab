import os
import numpy as np


def save_tenfold_weights(params: dict, weights_list: list, output_dir: str) -> None:
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    # パラメータ文字列を生成
    param_str = f"Nx{params['Nx']}_dens{params['density']}_inscl{params['input_scale']}_rho{params['rho']}"
    
    # 各foldの重みを個別に保存
    for i, weight in enumerate(weights_list):
        filename = f"weights_fold{i}_{param_str}.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez(filepath, weight=weight)
