import numpy as np

from esn_lab.model.esn import ESN


def pred(model: ESN, id, U, D):
    model.Reservoir.reset_reservoir_state()

    pred_len = len(U)
    Y_pred = []
    
    # 時間発展
    for n in range(pred_len):
        x_in = model.Input(U[n])

        # リザバー状態ベクトル
        x = model.Reservoir(x_in)

        # 学習後のモデル出力
        y_pred = model.Output(x)
        Y_pred.append(model.output_func(y_pred))

    return {
        "id": id,
        "predictions": np.array(Y_pred),
        "labels": np.array(D),
    } 
