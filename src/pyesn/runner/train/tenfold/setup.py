import os
import random
import numpy as np

def init_global_worker_env():
    """
    並列処理のプロセス全体で一度だけ呼ばれる初期化関数。
    各種ライブラリのスレッド数を1に制限し、競合を防ぐ。
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

def setup_worker_seed(seed: int):
    """
    各ワーカープロセスで呼ばれる初期化関数。
    乱数シードを設定して再現性を担保する。
    """
    random.seed(seed)
    np.random.seed(seed)
    # try:
    #     import cupy as cp
    #     cp.random.seed(seed)
    # except ImportError:
    #     pass