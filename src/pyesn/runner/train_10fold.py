# runner/train.py
import os
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from pyesn.pipeline.trainer import Trainer
from pyesn.utils.data_processing import make_onehot
from pyesn.model.model_builder import get_model, get_model_param_str


"""
setup ============================================================================
"""

# ワーカ全体の初期化
def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass

# 各ワーカの初期化
def _worker_setup(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cupy as cp
        cp.random.seed(seed)
    except Exception:
        pass


"""
utils ============================================================================
"""

def _load_10fold_csvs(csv_dir: Path) -> dict[str, Path]:
    letters = [chr(c) for c in range(ord("a"), ord("j") + 1)]
    mapping: dict[str, Path] = {}
    for ch in letters:
        p = csv_dir / f"10fold_{ch}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        mapping[ch] = p
    return mapping


def _read_pairs(paths: list[Path]) -> tuple[list[str], list[str], list[int]]:
    ids: list[str] = []
    img_paths: list[str] = []
    class_ids: list[int] = []

    usecols = ["file_path", "behavior"]
    for p in paths:
        df = pd.read_csv(p, usecols=usecols)
        # 厳密に列名を固定（勝手な補完はしない）
        if list(df.columns) != usecols:
            raise ValueError(f"CSV columns mismatch at {p}. expected={usecols}, got={list(df.columns)}")
        for fp, beh in zip(df["file_path"], df["behavior"]):
            fp = str(fp)
            ids.append(Path(fp).stem)
            img_paths.append(fp)
            class_ids.append(int(beh))
    return ids, img_paths, class_ids



def _flatten_search_space(ss: dict[str, list] | None) -> list[tuple[dict, str]]:
    """
    {"model.Nx":[10,20], "model.rho":[0.8,0.9]} →
    [({"Nx":10,"rho":0.8},"Nx=10_rho=0.8"), ...] のリストに変換。
    """
    if not ss:
        return [({}, "default")]
    items = []
    for k, vals in ss.items():
        if not k.startswith("model."):
            raise ValueError(f"search_space key must start with 'model.': {k}")
        field = k.split(".", 1)[1]
        items.append((field, list(vals)))
    # 直積
    keys = [k for k,_ in items]
    lists = [v for _,v in items]
    combos = []
    for values in product(*lists):
        d = {k:v for k,v in zip(keys, values)}
        tag = "_".join([f"{k}={v}" for k,v in d.items()])
        combos.append((d, tag))
    return combos




"""
fold&search ============================================================================
"""

def _run_one_fold_search(cfg, table, letters, leave, hp_overrides: dict, hp_tag: str, weight_dir: str):
    idx = letters.index(leave)
    _worker_setup(idx)

    train_letters = [x for x in letters if x != leave]
    tag = "".join(train_letters)  # 例: "bcdefghij"

    print(f"[INFO] 10fold train (search: {hp_tag}): use={train_letters} (leave_out='{leave}')")

    # 3-1) データの読み込み
    csv_paths = [table[ch] for ch in train_letters]
    ids, paths, class_ids = _read_pairs(csv_paths)
    assert len(ids) == len(paths) == len(class_ids), "length mismatch"

    # 3-2) モデル・最適化器・トレーナ初期化（差し替え部分）
    trainer = Trainer(cfg.run_dir)
    model, optimizer = get_model(cfg, hp_overrides)

    # 3-3) 学習ループ（既存と同じ）
    Ny = cfg.model.Ny
    for i in range(len(paths)):
        img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {paths[i]}")
        U = img.T
        T = len(U)
        D = make_onehot(class_ids[i], T, Ny)
        trainer.train(model, optimizer, ids[i], U, D)

    trainer.save_output_weight(Wout=model.Output.Wout, 
                               filename=f"{get_model_param_str(cfg=cfg, overrides=hp_overrides)}_{tag}_Wout.npy", 
                               save_dir=weight_dir)

    return 



def tenfold_train(cfg, *, parallel: bool = True, max_workers: int = 10):
    tenfold_cfg = cfg.train.tenfold_search
    if tenfold_cfg is None:
        raise ValueError("cfg.train.tenfold_search が見つかりません（tenfold_search.yaml の読み込み位置を確認してください）")

    csv_dir = Path(tenfold_cfg.csv_dir).expanduser().resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")

    # 検索空間の展開
    combos = _flatten_search_space(getattr(tenfold_cfg, "search_space", None))
    print(f"combos:{combos}\n\n")

    # 10fold 材料は一度作って使い回す
    table = _load_10fold_csvs(csv_dir)
    letters = sorted(table.keys())  # [a~j]

    weight_dir = f"{os.getcwd()}/{tenfold_cfg.weight_path}"

    # 並列器の設定は既存 tenfold_train と同じ方針
    workers = min(max_workers, (os.cpu_count() or max_workers))
    if os.name == "posix":
        mp_ctx = mp.get_context("fork")
        executor_kwargs = {"max_workers": workers, "mp_context": mp_ctx}
    else:
        executor_kwargs = {"max_workers": workers}

    # ハイパラ組ループ
    for hp_overrides, hp_tag in combos:
        print("\n=====================================")
        print(f"[INFO] hyperparam combo: {hp_tag}")
        print("=====================================")

        # 処理済みタスクかのフラグ
        tasks_to_run = []

        # 1. 全てのfoldをチェックし、実行タスクとスキップタスクを振り分ける
        # foldループ
        for leave in letters:
            train_letters = [x for x in letters if x != leave]
            tag = "".join(train_letters)

            weight_filename = f"{get_model_param_str(cfg=cfg, overrides=hp_overrides)}_{tag}_Wout.npy"
            expected_path = f"{weight_dir}/{weight_filename}"
            print(expected_path)

            # 重みファイルが存在しない場合、実行リストに追加
            if expected_path.exists():
                print(f"[INFO] Weight file found, skipping fold '{leave}': {expected_path}")
            else:
                tasks_to_run.append(leave)

        # 探索済みの場合スキップ
        if not tasks_to_run:
            continue

        # 実行部分
        # 並列がfalseの場合逐次処理
        if not parallel:    
            for leave in tasks_to_run:
                _run_one_fold_search(cfg, table, letters, leave, hp_overrides, hp_tag, weight_dir)
        # 並列実行パターン
        else:  
            with ProcessPoolExecutor(**executor_kwargs, initializer=_init_worker) as ex:
                future_to_leave = {
                    ex.submit(_run_one_fold_search, cfg, table, letters, leave, hp_overrides, hp_tag, weight_dir): leave
                    for leave in tasks_to_run
                }

                for future in as_completed(future_to_leave):
                    leave = future_to_leave[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Fold '{leave}' in combo '{hp_tag}' failed: {e}")
            

    print("=====================================")
    print("[INFO] tenfold hyperparameter search finished.")
    print("=====================================")
    return