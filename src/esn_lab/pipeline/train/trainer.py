# pipeline/trainer
import numpy as np

from esn_lab.model.esn import ESN

def train(model: ESN, optimizer, U_list, D_list):

    for U, D in zip(U_list, D_list): 
        train_len = len(U)

        # 時間発展
        for n in range(train_len):
            x_in = model.Input(U[n])

            # リザバー状態ベクトル
            x = model.Reservoir(x_in)

            # 目標値
            d = D[n]
            d = model.inv_output_func(d)

            # 学習器
            if n > 0:  
                optimizer(d, x)     # 1データあたりの学習結果が逐次optimizerに記憶されていく

        model.Reservoir.x = np.zeros(model.N_x)     # リザバー状態のリセット

    output_weight = optimizer.get_Wout_opt()

    return output_weight

