# pipeline/trainer
import json
import numpy as np
from pathlib import Path

from mypkg.utils.constants import Params
from mypkg.model.esn import ESN
from mypkg.utils.config import TargetOutput, TargetOutputData
from mypkg.utils.io import to_jsonable


class Trainer:
    def __init__(self, run_dir):
        self.params = Params()
        self.output_weight_dir = Path(run_dir + "/output_weight")
        self.output_weight_dir.mkdir(parents=True, exist_ok=True)


    
    def train(self, model: ESN, optimizer, sample_id, U, D, trans_len = None):
        if trans_len is None:
            trans_len = 0
        Y = []
        D_save = []
        train_len = len(U)

        # 時間発展
        for n in range(train_len):
            x_in = model.Input(U[n])

            # フィードバック結合
            if model.Feedback is not None:
                x_back = model.Feedback(model.y_prev)
                x_in += x_back
            # ノイズ
            if model.noise is not None:
                x_in += model.noise

            # リザバー状態ベクトル
            x = model.Reservoir(x_in)

            # 分類問題の場合は窓幅分の平均を取得
            if model.classification:
                model.window = np.append(model.window, x.reshape(1, -1),
                                        axis=0)
                model.window = np.delete(model.window, 0, 0)
                x = np.average(model.window, axis=0)

            # 目標値
            d = D[n]
            d = model.inv_output_func(d)
            D_save.append(d)

            # 学習器
            if n > trans_len:  
                optimizer(d, x)     # 1データあたりの学習結果が逐次optimizerに記憶されていく

            # 学習前のモデル出力
            y = model.Output(x)
            Y.append(model.output_func(y))
            model.y_prev = d


        # 学習前モデル出力と教師ラベルの記憶
        Y = np.array(Y)
        D_save = np.array(D_save)
        
        model.Output.setweight(optimizer.get_Wout_opt())    # 学習済みの出力結合重み行列を設定
        model.Reservoir.x = np.zeros(model.N_x)     # リザバー状態のリセット
        print(f"[INFO] {sample_id} is trained")

        data = TargetOutputData(
            target_series= to_jsonable(D_save),
            output_series= to_jsonable(Y),
        )
        result = TargetOutput(
            id= sample_id,
            data= data
        )

        # return {sample_id : {TARGET_SERIES_KEY: D_save.item(), OUTPUT_SERIES_KEY: Y}}
        return result


    def save_output_weight(self, model: ESN):
        weight_name = model.get_param_list() + "_Wout.npy"
        np.save(Path(self.output_weight_dir) / Path(weight_name), model.Output.Wout)
        print("[INFO] output weight saved\n=====================================")
        return
    