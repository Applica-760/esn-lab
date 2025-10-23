# pipeline/evaluator.py
import cv2
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from pyesn.setup.config import TargetOutput, Config
from pyesn.model.esn import ESN
from pyesn.model.model_builder import get_model, get_model_param_str
from pyesn.pipeline.predictor import Predictor
from pyesn.utils.data_processing import make_onehot



class Evaluator:

    # 毎時刻評価
    def evaluate_classification_result(self, record: TargetOutput):
        Y = np.array(record.data.output_series)
        D = np.array(record.data.target_series)
        pred_idx = Y.argmax(axis=1) # モデル出力の最大値インデックス番号を取得
        T = D.shape[0]  # 時系列長を取得

        # モデル予測と正解ラベルが一致していたらTrue
        correct_mask = D[np.arange(T), pred_idx].astype(bool)   

        num_correct = int(correct_mask.sum())
        acc = float(correct_mask.mean())
        return num_correct, acc, correct_mask
    

    # 従来多数決評価
    def majority_success(self, record: TargetOutput):
        Y = np.array(record.data.output_series)
        D = np.array(record.data.target_series)
        pred_idx = Y.argmax(axis=1)    
        true_idx = D.argmax(axis=1)     

        C = D.shape[1]
        pred_counts = np.bincount(pred_idx, minlength=C)
        true_counts = np.bincount(true_idx, minlength=C)

        pred_major = int(pred_counts.argmax())
        true_major = int(true_counts.argmax())

        success = (pred_major == true_major)
        return success, pred_major, true_major, pred_counts, true_counts


    def make_confusion_matrix(self):
        return

