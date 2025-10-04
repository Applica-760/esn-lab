# runner/train.py
import cv2
import numpy as np

from mypkg.utils.config import Config
from mypkg.pipeline.trainer import Trainer
from mypkg.model.model_builder import get_model
from mypkg.utils.io import save_json, to_keyed_dict
from mypkg.utils.constants import TRAIN_RECORD_FILE


def make_onehot(class_id: int, T: int, num_of_class: int) -> np.ndarray:
    return np.tile(np.eye(num_of_class)[class_id], (T, 1))


def single_train(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)
    
    # load data
    U = cv2.imread(cfg.train.single.path, cv2.IMREAD_UNCHANGED).T
    train_len = len(U)
    D = make_onehot(cfg.train.single.class_id, train_len, cfg.model.Ny)

    # set trainer
    trainer = Trainer(cfg.run_dir)
    result = trainer.train(model, optimizer, cfg.train.single.id, U, D)
    print("=====================================")

    # log output layer
    save_json(to_keyed_dict(result), cfg.run_dir, TRAIN_RECORD_FILE)
    print("[INFO] output layer saved to json\n=====================================")

    # save output weight
    trainer.save_output_weight(model)

    return result


def batch_train(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)
    
    trainer = Trainer(cfg.run_dir)
    results = {}
    for i in range(len(cfg.train.batch.ids)):
        U = cv2.imread(cfg.train.batch.paths[i], cv2.IMREAD_UNCHANGED).T
        train_len = len(U)
        D = make_onehot(cfg.train.batch.class_ids[i], train_len, cfg.model.Ny)
        result = to_keyed_dict(trainer.train(model, optimizer, cfg.train.batch.ids[i], U, D))
        results.update(result)
    
    print("=====================================")
    # log output layer
    save_json(results, cfg.run_dir, TRAIN_RECORD_FILE)
    print("[INFO] output layer saved to json\n=====================================")

    # save output weight
    trainer.save_output_weight(model)

    return results