# model/esn.py
import numpy as np
import networkx as nx

from esn_lab.utils.activate_func import identity


# 入力層
class Input:
    def __init__(self, N_u, N_x, input_scale, seed=0):
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    def __call__(self, u):
        return np.dot(self.Win, u)


# リザバー
class Reservoir:
    def __init__(self, N_x, density, rho, activation_func, leaking_rate, seed=0):
        self.seed = seed
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)  
        self.activation_func = activation_func
        self.alpha = leaking_rate

    def make_connection(self, N_x, density, rho):
        m = int(N_x*(N_x-1)*density/2) 
        G = nx.gnm_random_graph(N_x, m, self.seed)

        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        W *= float(rho) / float(sp_radius)  
        return W

    def __call__(self, x_in):
        #self.x = self.x.reshape(-1, 1)
        self.x = (1.0 - self.alpha) * self.x \
                 + self.alpha * self.activation_func(np.dot(self.W, self.x) \
                 + x_in)
        return self.x

    def reset_reservoir_state(self):
        self.x *= 0.0


# 出力層
class Output:
    def __init__(self, N_x, N_y, seed=0):
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_x))

    def __call__(self, x):
        return np.dot(self.Wout, x)

    def setweight(self, Wout_opt):
        self.Wout = Wout_opt


# エコーステートネットワーク
class ESN:
    # 各層の初期化
    def __init__(self, N_u, N_y, N_x, density, input_scale,
                 rho, activation_func=np.tanh, leaking_rate=1.0,
                 output_func=identity, inv_output_func=identity):
        
        self.Input = Input(N_u, N_x, input_scale)
        self.Reservoir = Reservoir(N_x, density, rho, activation_func, 
                                   leaking_rate)
        self.Output = Output(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x

        # 追加部分
        self.density = density
        self.input_scale = input_scale
        self.rho = rho

        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
