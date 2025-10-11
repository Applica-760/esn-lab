# runner/train.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

from pyesn.utils.config import Config
from pyesn.pipeline.trainer import Trainer
from pyesn.utils.data_processing import make_onehot
from pyesn.model.model_builder import get_model, get_model_param_str


def single_train(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)
    
    # load data
    U = cv2.imread(cfg.train.single.path, cv2.IMREAD_UNCHANGED).T
    train_len = len(U)
    D = make_onehot(cfg.train.single.class_id, train_len, cfg.model.Ny)

    # set trainer
    trainer = Trainer(cfg.run_dir)
    trainer.train(model, optimizer, cfg.train.single.id, U, D)
    print("=====================================")

    # save output weight
    trainer.save_output_weight(model.Output.Wout, f"{get_model_param_str(cfg=cfg)}_Wout.npy")

    return


def batch_train(cfg: Config):
    # model定義
    model, optimizer = get_model(cfg)
    
    trainer = Trainer(cfg.run_dir)
    for i in range(len(cfg.train.batch.ids)):
        U = cv2.imread(cfg.train.batch.paths[i], cv2.IMREAD_UNCHANGED).T
        train_len = len(U)
        D = make_onehot(cfg.train.batch.class_ids[i], train_len, cfg.model.Ny)
        trainer.train(model, optimizer, cfg.train.batch.ids[i], U, D)

    # save output weight
    trainer.save_output_weight(model.Output.Wout, f"{get_model_param_str(cfg=cfg)}_Wout.npy")

    return


# # 10fold -----------------------------------------------------------------------------------------
"""
重複定義を避け、`train_10fold.py`に一元化したい。
"""


# def _load_10fold_csvs(csv_dir: Path) -> dict[str, Path]:
#     """csv_dir から 10fold_{a..j}.csv を集める。全10枚なければエラー。"""
#     letters = [chr(c) for c in range(ord("a"), ord("j") + 1)]
#     mapping: dict[str, Path] = {}
#     for ch in letters:
#         p = csv_dir / f"10fold_{ch}.csv"
#         if not p.exists():
#             raise FileNotFoundError(f"missing: {p}")
#         mapping[ch] = p
#     return mapping


# def _read_pairs(paths: list[Path]) -> tuple[list[str], list[str], list[int]]:
#     """CSV 群から (ids, image_paths, class_ids) を作る。"""
#     ids: list[str] = []
#     img_paths: list[str] = []
#     class_ids: list[int] = []

#     usecols = ["file_path", "behavior"]
#     for p in paths:
#         df = pd.read_csv(p, usecols=usecols)
#         # 厳密に列名を固定（勝手な補完はしない）
#         if list(df.columns) != usecols:
#             raise ValueError(f"CSV columns mismatch at {p}. expected={usecols}, got={list(df.columns)}")
#         for fp, beh in zip(df["file_path"], df["behavior"]):
#             fp = str(fp)
#             ids.append(Path(fp).stem)
#             img_paths.append(fp)
#             class_ids.append(int(beh))
#     return ids, img_paths, class_ids

# # ワーカ全体の初期化
# def _init_worker():
#     os.environ.setdefault("OMP_NUM_THREADS", "1")
#     os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#     os.environ.setdefault("MKL_NUM_THREADS", "1")
#     os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#     try:
#         import cv2
#         cv2.setNumThreads(1)
#     except Exception:
#         pass

# # 各ワーカの初期化
# def _worker_setup(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     try:
#         import cupy as cp
#         cp.random.seed(seed)
#     except Exception:
#         pass


# # 1foldあたりの実装
# def _run_one_fold(cfg, table, letters, leave):
#     idx = letters.index(leave)
#     _worker_setup(idx)

#     train_letters = [x for x in letters if x != leave]
#     tag = "".join(train_letters)  # 例: "bcdefghij"

#     print("\n=====================================")
#     print(f"[INFO] 10fold train: use={train_letters} (leave_out='{leave}')")
#     print("=====================================")

#     # 3-1) データの読み込み
#     csv_paths = [table[ch] for ch in train_letters]
#     ids, paths, class_ids = _read_pairs(csv_paths)
#     assert len(ids) == len(paths) == len(class_ids), "length mismatch"

#     # 3-2) モデル・最適化器・トレーナ初期化
#     trainer = Trainer(cfg.run_dir)
#     model, optimizer = get_model(cfg)

#     # 3-3) 学習ループ
#     Ny = cfg.model.Ny
#     for i in range(len(paths)):
#         img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
#         if img is None:
#             raise FileNotFoundError(f"failed to read image: {paths[i]}")
#         U = img.T
#         T = len(U)
#         D = make_onehot(class_ids[i], T, Ny)
#         trainer.train(model, optimizer, ids[i], U, D)

#     trainer.save_output_weight(Wout=model.Output.Wout, filename=tag)

#     return {
#         "tag" : tag,
#         "num_samples": len(paths),
#         "letters_used": train_letters,
#     }


# # オーケストレーション
# def tenfold_train(cfg, *, parallel: bool = True):
#     """
#     既存 tenfold_train の並列版。
#     cfg は cfg のまま子プロセスへ渡す。foldごとの材料は fold_payload に分離。
#     """
#     try:
#         csv_dir = Path(cfg.train.tenfold.csv_dir).expanduser().resolve()
#     except Exception as e:
#         raise ValueError("cfg.data['csv_dir'] が設定されていません。10fold_* CSV のあるディレクトリを指定してください。") from e
#     if not csv_dir.exists():
#         raise FileNotFoundError(f"csv_dir not found: {csv_dir}")
    
#     table = _load_10fold_csvs(csv_dir)
#     letters = sorted(table.keys())  # ["a", ..., "j"]

#     if not parallel:
#         all_results = {}
#         for leave in letters:
#             summary = _run_one_fold(cfg, table, letters, leave)
#             tag = summary.get("tag")
#             all_results[tag] = summary

#         return all_results

#     max_workers = cfg.train.tenfold.workers
#     workers = min(max_workers, (os.cpu_count() or max_workers))

#     if os.name == "posix":
#         mp_ctx = mp.get_context("fork")
#         executor_kwargs = {"max_workers": workers, "mp_context": mp_ctx}
#     else:
#         executor_kwargs = {"max_workers": workers}

#     all_results = {}
#     with ProcessPoolExecutor(**executor_kwargs, initializer=_init_worker) as ex:
#         futures = []
#         for leave in letters:
#             futures.append((leave, ex.submit(_run_one_fold, cfg, table, letters, leave)))

#         for leave, fut in futures:
#             summary = fut.result()
#             tag = summary["tag"]
#             all_results[tag] = summary

#     print("=====================================")
#     print("[INFO] 10fold training finished (parallel).")
#     print("=====================================")
#     return all_results


