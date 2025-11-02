# runner/train.py
import cv2

from esn_lab.setup.config import Config
from esn_lab.pipeline.train.trainer import Trainer
from esn_lab.utils.data_processing import make_onehot
from esn_lab.model.model_builder import get_model, get_model_param_str
from esn_lab.utils.io import save_numpy_npy_atomic


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

    # save output weight (utilに委譲)
    dst = save_numpy_npy_atomic(
        model.Output.Wout,
        trainer.output_weight_dir,
        f"{get_model_param_str(cfg=cfg)}_Wout.npy",
    )
    print(f"[INFO] Saved weight: {dst}")

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

    # save output weight (utilに委譲)
    dst = save_numpy_npy_atomic(
        model.Output.Wout,
        trainer.output_weight_dir,
        f"{get_model_param_str(cfg=cfg)}_Wout.npy",
    )
    print(f"[INFO] Saved weight: {dst}")

    return

