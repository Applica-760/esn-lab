# model/model_builder.py
from mypkg.model.esn import ESN
from mypkg.optim.optim import Tikhonov
from mypkg.utils.config import Config


def get_model(cfg: Config):
    model = ESN(cfg.model.Nu, cfg.model.Ny, cfg.model.Nx, cfg.model.density, cfg.model.input_scale, cfg.model.rho)
    optimizer = Tikhonov(cfg.model.Nx, cfg.model.Ny, 0.0)
    return model, optimizer

