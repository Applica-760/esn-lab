# pipeline/trainer
import json
import numpy as np
from pathlib import Path
import os

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


    def save_output_weight(self, model, filename: str | None = None):
        """
        学習済みWoutを保存するユーティリティ
        """
        # 出力先ディレクトリ
        out_dir = Path(self.output_weight_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = model.get_param_list()
        if filename is None:
            save_name = f"{base}_Wout.npy"
        else:
            save_name = f"{base}_{filename}_Wout.npy"

        dst = out_dir / save_name
        tmp = dst.with_suffix(dst.suffix + ".tmp") 

        # 実体は model.Output.Wout（従来どおり）
        np.save(tmp, model.Output.Wout)
        os.replace(tmp, dst) 
        if hasattr(self, "logger"):
            self.logger.info(f"Saved weight: {dst}")
        else:
            print(f"[INFO] Saved weight: {dst}")