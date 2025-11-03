# pipeline/predictor.py
import numpy as np
from pathlib import Path

from esn_lab.model.esn import ESN
from esn_lab.setup.config import TargetOutput, TargetOutputData
from esn_lab.utils.io import to_jsonable


class Predictor:
    def __init__(self, run_dir):
        self.predict_result = Path(run_dir + "/predict_result")
        self.predict_result.mkdir(parents=True, exist_ok=True)


    def predict(self, model:ESN, sample_id, U, D):
        model.Reservoir.reset_reservoir_state()
        model.y_prev = np.zeros(model.N_y)

        test_len = len(U)
        Y_pred = []
        # 時間発展
        for n in range(test_len):
            x_in = model.Input(U[n])

            # リザバー状態ベクトル
            x = model.Reservoir(x_in)

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

        return result
