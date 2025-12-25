from itertools import product


def build_param_grid(cfg):
    param_names = ['Nx', 'input_scale', 'density', 'rho']
    param_values = [cfg.Nx, cfg.input_scale, cfg.density, cfg.rho]
    
    # 全組み合わせを生成
    combinations = product(*param_values)

    print("grid params built")
    
    # 辞書のリストに変換
    return [dict(zip(param_names, combo)) for combo in combinations]
