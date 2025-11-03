# runner/predict.py
import cv2
import numpy as np
from pathlib import Path

from esn_lab.setup.config import Config
from esn_lab.pipeline.pred.predictor import Predictor
from esn_lab.model.model_builder import get_model
from esn_lab.utils.data_processing import make_onehot
from esn_lab.utils.io import save_json, to_keyed_dict
from esn_lab.utils.constants import PREDICT_RECORD_FILE


def single_predict(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)

    # load weight
    weight_dir = Path(cfg.predict.single.weight)
    weight_path = sorted(weight_dir.glob("*.npy"))[0]
    weight = np.load(weight_path, allow_pickle=True)
    model.Output.setweight(weight)
    print(f"[ARTIFACT] {weight_path} is loaded")

    # load data
    U = cv2.imread(cfg.predict.single.path, cv2.IMREAD_UNCHANGED).T
    predict_len = len(U)
    D = make_onehot(cfg.predict.single.class_id, predict_len, cfg.model.Ny)

    # set predictor
    predictor = Predictor(cfg.run_dir)
    result = predictor.predict(model, cfg.predict.single.id, U, D)
    print("=====================================")

    # log output layer
    save_json(to_keyed_dict(result), cfg.run_dir, PREDICT_RECORD_FILE)
    print("[INFO] output layer saved to json\n=====================================")

    return


def batch_predict(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)

    # load weight
    weight_dir = Path(cfg.predict.batch.weight)
    weight_path = sorted(weight_dir.glob("*.npy"))[0]
    weight = np.load(weight_path)
    model.Output.setweight(weight)
    print(f"[ARTIFACT] {weight_path} is loaded")

    # set predictor
    predictor = Predictor(cfg.run_dir)
    results = {}
    for i in range(len(cfg.predict.batch.ids)):
        U = cv2.imread(cfg.predict.batch.paths[i], cv2.IMREAD_UNCHANGED).T
        predict_len = len(U)
        D = make_onehot(cfg.predict.batch.class_ids[i], predict_len, cfg.model.Ny)
        result = to_keyed_dict(predictor.predict(model, cfg.predict.batch.ids[i], U, D))
        results.update(result)

    print("=====================================")

    # log output layer
    save_json(results, cfg.run_dir, PREDICT_RECORD_FILE)
    print("[INFO] output layer saved to json\n=====================================")

    return


# 重みを変えてループ回しながら，batch_predictionを呼び出して