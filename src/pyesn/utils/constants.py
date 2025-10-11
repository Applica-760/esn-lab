import cv2
import numpy as np
from pathlib import Path

TRAIN_RECORD_FILE = "train_record.jsonl"
PREDICT_RECORD_FILE = "predict_record.jsonl"

TARGET_SERIES_KEY = "target_series"
OUTPUT_SERIES_KEY = "output_series"


class Params:
    def __init__(self):
        self.NUM_OF_CLASS = 3
        self.path_to_prj = Path(__file__).resolve().parents[2]

        self.num2behavior = {0: 'other',
                             1: 'foraging',
                             2: 'rumination'}
        self.behavior2num = {'other': 0,
                             'foraging': 1,
                             'rumination': 2}


class SampleInfo:
    def __init__(self, id, path, class_id):
        self.id = id
        self.path = path
        self.class_id = class_id


class Histogram:
    def __init__(self, id, behavior, path):
        self.id = id
        self.behavior = behavior
        img = cv2.imread(path + id + '.jpg', cv2.IMREAD_UNCHANGED)
        self.img = img.T
        T = len(self.img)

        # 酪農学園大学教師データに基づくやつ
        behavior2teachingdata = {
            'other': np.hstack([np.ones((T, 1)), np.zeros((T, 2))]),
            'foraging': np.hstack([np.zeros((T, 1)), np.ones((T, 1)), np.zeros((T, 1))]),
            'rumination': np.hstack([np.zeros((T, 2)), np.ones((T, 1))])
        }

        self.teachingData = behavior2teachingdata[behavior]
        print(self.id + ' is loaded')



