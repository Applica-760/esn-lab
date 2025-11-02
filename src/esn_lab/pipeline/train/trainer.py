# pipeline/trainer
import numpy as np
from pathlib import Path

from esn_lab.model.esn import ESN


class Trainer:
    def __init__(self, run_dir):
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

            # リザバー状態ベクトル
            x = model.Reservoir(x_in)

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

        return

