# pipeline/predictor.py
import json
import numpy as np
from pathlib import Path

from pyesn import ESN
from pyesn.utils.constants import Params
from pyesn.utils.config import TargetOutput, TargetOutputData
from pyesn.utils.io import to_jsonable


class Predictor:
    def __init__(self, run_dir):
        self.params = Params()
        self.predict_result = Path(run_dir + "/predict_result")
        self.predict_result.mkdir(parents=True, exist_ok=True)


    def predict(self, model:ESN, sample_id, U, D):
        test_len = len(U)
        Y_pred = []
        behavior_count = []
        # 時間発展
        for n in range(test_len):
            x_in = model.Input(U[n])

            # フィードバック結合
            if model.Feedback is not None:
                x_back = model.Feedback(model.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = model.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if model.classification:
                model.window = np.append(model.window, x.reshape(1, -1),
                                        axis=0)
                model.window = np.delete(model.window, 0, 0)
                x = np.average(model.window, axis=0)

            # 学習後のモデル出力
            y_pred = model.Output(x)
            Y_pred.append(model.output_func(y_pred))
            model.y_prev = y_pred

        print(f"[INFO] {sample_id} is predicted")

        data = TargetOutputData(
            target_series= to_jsonable(D),
            output_series= to_jsonable(np.array(Y_pred)),
        )
        result = TargetOutput(
            id= sample_id,
            data= data
        )
        
        # return {sample_id : {TARGET_SERIES_KEY: D, OUTPUT_SERIES_KEY: np.array(Y_pred)}}
        return result
