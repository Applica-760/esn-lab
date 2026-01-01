# esn_lab/__init__.py

from esn_lab.model.esn import ESN
from esn_lab.optim.optim import Tikhonov
from esn_lab.pipeline.train.trainer import train
from esn_lab.runner.train.tenfold import run_tenfold
from esn_lab.runner.train.tenfold_parallel import run_tenfold_parallel


__all__ = [
    "ESN",
    "Tikhonov",
    "train",
    "run_tenfold",
    "run_tenfold_parallel",
]

__version__ = "0.1.0"
