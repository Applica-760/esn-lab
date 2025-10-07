# model/model_builder.py
from mypkg.model.esn import ESN
from mypkg.optim.optim import Tikhonov
from mypkg.utils.config import Config


def get_model(cfg: Config):
    model = ESN(cfg.model.Nu, cfg.model.Ny, cfg.model.Nx, cfg.model.density, cfg.model.input_scale, cfg.model.rho)
    optimizer = Tikhonov(cfg.model.Nx, cfg.model.Ny, 0.0)
    return model, optimizer


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