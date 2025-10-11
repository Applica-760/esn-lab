# runner/train.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

from pyesn.setup.config import Config
from pyesn.pipeline.trainer import Trainer
from pyesn.utils.data_processing import make_onehot
from pyesn.model.model_builder import get_model, get_model_param_str


def single_train(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)
    
    # load data
    U = cv2.imread(cfg.train.single.path, cv2.IMREAD_UNCHANGED).T
    train_len = len(U)
    D = make_onehot(cfg.train.single.class_id, train_len, cfg.model.Ny)

    # set trainer
    trainer = Trainer(cfg.run_dir)
    trainer.train(model, optimizer, cfg.train.single.id, U, D)
    print("=====================================")

    # save output weight
    trainer.save_output_weight(model.Output.Wout, f"{get_model_param_str(cfg=cfg)}_Wout.npy")

    return


def batch_train(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)
    
    trainer = Trainer(cfg.run_dir)
    for i in range(len(cfg.train.batch.ids)):
        U = cv2.imread(cfg.train.batch.paths[i], cv2.IMREAD_UNCHANGED).T
        train_len = len(U)
        D = make_onehot(cfg.train.batch.class_ids[i], train_len, cfg.model.Ny)
        trainer.train(model, optimizer, cfg.train.batch.ids[i], U, D)

    # save output weight
    trainer.save_output_weight(model.Output.Wout, f"{get_model_param_str(cfg=cfg)}_Wout.npy")

    return

