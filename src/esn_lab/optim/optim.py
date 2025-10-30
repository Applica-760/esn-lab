import numpy as np


# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_x))
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):

        A = self.X_XT + self.beta * np.identity(self.N_x, dtype=self.X_XT.dtype)
        B = self.D_XT.T

        try:
            # 通常ルート（可逆行列の場合）
            Wout_opt = np.linalg.solve(A, B).T

        except np.linalg.LinAlgError:
            # 特異行列などで解けなかった場合のフォールバック
            X_pseudo_inv = np.linalg.pinv(self.X_XT + self.beta * np.identity(self.N_x))
            Wout_opt = np.dot(self.D_XT, X_pseudo_inv)

        return Wout_opt
