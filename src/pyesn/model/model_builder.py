# model/model_builder.py
from pyesn.model.esn import ESN
from pyesn.optim.optim import Tikhonov
from pyesn.setup.config import Config

# 基本のモデル取得メソッド
def get_model(cfg: Config):
    model = ESN(cfg.model.Nu, cfg.model.Ny, cfg.model.Nx, cfg.model.density, cfg.model.input_scale, cfg.model.rho)
    optimizer = Tikhonov(cfg.model.Nx, cfg.model.Ny, 0.0)
    return model, optimizer

# 10foldパラメタサーチ対応
def get_model_with_overrides(cfg: Config, overrides: dict):
    """
    cfg.model の値を overrides で部分的に上書きしてモデル・最適化器を返す。
    """
    # 既存を壊さないよう、cfg はコピーして読み出しのみ
    Nu = overrides.get("Nu", cfg.model.Nu)
    Nx = overrides.get("Nx", cfg.model.Nx)
    Ny = overrides.get("Ny", cfg.model.Ny)
    density = overrides.get("density", cfg.model.density)
    input_scale = overrides.get("input_scale", cfg.model.input_scale)
    rho = overrides.get("rho", cfg.model.rho)

    model = ESN(Nu, Ny, Nx, density, input_scale, rho)
    optimizer = Tikhonov(Nx, Ny, 0.0)  
    return model, optimizer

# weight保存に使用するパラメタ取得へルパ
def get_model_param_str(cfg: Config, overrides: dict | None = None, seed_id="nonseed") -> str:
    if overrides is None:
        overrides = {}
        
    N_x = overrides.get("Nx", cfg.model.Nx)
    density = overrides.get("density", cfg.model.density)
    input_scale = overrides.get("input_scale", cfg.model.input_scale)
    rho = overrides.get("rho", cfg.model.rho)

    param_list = (f"seed-{seed_id}"
                  f"_nx-{N_x}"
                  f"_density-{density}"
                  f"_input_scale-{input_scale}"
                  f"_rho-{rho}").replace(".", "")
    return param_list