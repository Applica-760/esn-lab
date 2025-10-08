# runner/train.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os


from mypkg.utils.config import Config
from mypkg.pipeline.trainer import Trainer
from mypkg.model.model_builder import get_model, get_model_with_overrides 
from mypkg.utils.io import save_json, to_keyed_dict
from mypkg.utils.constants import TRAIN_RECORD_FILE


def make_onehot(class_id: int, T: int, num_of_class: int) -> np.ndarray:
    return np.tile(np.eye(num_of_class)[class_id], (T, 1))


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
    trainer.save_output_weight(model)

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
    trainer.save_output_weight(model)

    return


# 10fold -----------------------------------------------------------------------------------------

def _load_10fold_csvs(csv_dir: Path) -> dict[str, Path]:
    """csv_dir から 10fold_{a..j}.csv を集める。全10枚なければエラー。"""
    letters = [chr(c) for c in range(ord("a"), ord("j") + 1)]
    mapping: dict[str, Path] = {}
    for ch in letters:
        p = csv_dir / f"10fold_{ch}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        mapping[ch] = p
    return mapping


def _read_pairs(paths: list[Path]) -> tuple[list[str], list[str], list[int]]:
    """CSV 群から (ids, image_paths, class_ids) を作る。"""
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


# 1foldあたりの実装
def _run_one_fold(cfg, table, letters, leave):
    idx = letters.index(leave)
    _worker_setup(idx)

    train_letters = [x for x in letters if x != leave]
    tag = "".join(train_letters)  # 例: "bcdefghij"

    print("\n=====================================")
    print(f"[INFO] 10fold train: use={train_letters} (leave_out='{leave}')")
    print("=====================================")

    # 3-1) データの読み込み
    csv_paths = [table[ch] for ch in train_letters]
    ids, paths, class_ids = _read_pairs(csv_paths)
    assert len(ids) == len(paths) == len(class_ids), "length mismatch"

    # 3-2) モデル・最適化器・トレーナ初期化
    trainer = Trainer(cfg.run_dir)
    model, optimizer = get_model(cfg)

    # 3-3) 学習ループ
    Ny = cfg.model.Ny
    for i in range(len(paths)):
        img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {paths[i]}")
        U = img.T
        T = len(U)
        D = make_onehot(class_ids[i], T, Ny)
        trainer.train(model, optimizer, ids[i], U, D)

    trainer.save_output_weight(model, filename=tag)

    return {
        "tag" : tag,
        "num_samples": len(paths),
        "letters_used": train_letters,
    }


# オーケストレーション
def tenfold_train(cfg, *, parallel: bool = True, max_workers: int = 10):
    """
    既存 tenfold_train の並列版。
    cfg は cfg のまま子プロセスへ渡す。foldごとの材料は fold_payload に分離。
    """
    try:
        csv_dir = Path(cfg.train.tenfold.csv_dir).expanduser().resolve()
    except Exception as e:
        raise ValueError("cfg.data['csv_dir'] が設定されていません。10fold_* CSV のあるディレクトリを指定してください。") from e
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")
    
    table = _load_10fold_csvs(csv_dir)
    letters = sorted(table.keys())  # ["a", ..., "j"]

    if not parallel:
        all_results = {}
        for leave in letters:
            summary = _run_one_fold(cfg, table, letters, leave)
            tag = summary.get("tag")
            all_results[tag] = summary

        return all_results

    workers = min(max_workers, (os.cpu_count() or max_workers))

    if os.name == "posix":
        mp_ctx = mp.get_context("fork")
        executor_kwargs = {"max_workers": workers, "mp_context": mp_ctx}
    else:
        executor_kwargs = {"max_workers": workers}

    all_results = {}
    with ProcessPoolExecutor(**executor_kwargs, initializer=_init_worker) as ex:
        futures = []
        for leave in letters:
            futures.append((leave, ex.submit(_run_one_fold, cfg, table, letters, leave)))

        for leave, fut in futures:
            summary = fut.result()
            tag = summary["tag"]
            all_results[tag] = summary

    print("=====================================")
    print("[INFO] 10fold training finished (parallel).")
    print("=====================================")
    return all_results



# 10fold search -----------------------------------------------------------------------------------------

def _run_one_fold_search(cfg, table, letters, leave, hp_overrides: dict, hp_tag: str):
    idx = letters.index(leave)
    _worker_setup(idx)

    train_letters = [x for x in letters if x != leave]
    tag = "".join(train_letters)  # 例: "bcdefghij"

    print("\n=====================================")
    print(f"[INFO] 10fold train (search: {hp_tag}): use={train_letters} (leave_out='{leave}')")
    print("=====================================")

    # 3-1) データの読み込み
    csv_paths = [table[ch] for ch in train_letters]
    ids, paths, class_ids = _read_pairs(csv_paths)
    assert len(ids) == len(paths) == len(class_ids), "length mismatch"

    # 3-2) モデル・最適化器・トレーナ初期化（差し替え部分）
    trainer = Trainer(cfg.run_dir)
    model, optimizer = get_model_with_overrides(cfg, hp_overrides)

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

    trainer.save_output_weight(model, filename=f"{hp_tag}_{tag}")

    return {
        "hp_tag": hp_tag,
        "tag" : tag,
        "num_samples": len(paths),
        "letters_used": train_letters,
    }

from itertools import product

def _flatten_search_space(ss: dict[str, list] | None) -> list[tuple[dict, str]]:
    """
    {"model.Nx":[10,20], "model.rho":[0.8,0.9]} →
    [({"Nx":10,"rho":0.8},"Nx=10_rho=0.8"), ...] のリストに変換。
    ※ model. 以外の prefix が来たときは弾く（勝手な拡張を避ける）
    """
    if not ss:
        return [({}, "default")]
    # model.* のみ許可（安全策）
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

def tenfold_search_train(cfg, *, parallel: bool = True, max_workers: int = 10):
    """
    ハイパラ組合せ（外側） × 10fold（内側）
    既存 tenfold_train を参考に、_run_one_fold_search() を使う。
    """
    # tenfold 設定の場所は既存に合わせる（cfg.train.tenfold 下）
    tenfold_cfg = cfg.train.tenfold_search
    if tenfold_cfg is None:
        raise ValueError("cfg.train.tenfold_search が見つかりません（tenfold_search.yaml の読み込み位置を確認してください）")

    csv_dir = Path(tenfold_cfg.csv_dir).expanduser().resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")

    # 検索空間の展開（B方式）
    combos = _flatten_search_space(getattr(tenfold_cfg, "search_space", None))
    print(f"combos:{combos}\n\n")

    # 10fold 材料は一度作って使い回す
    table = _load_10fold_csvs(csv_dir)
    letters = sorted(table.keys())

    # 並列器の設定は既存 tenfold_train と同じ方針
    workers = min(max_workers, (os.cpu_count() or max_workers))
    if os.name == "posix":
        mp_ctx = mp.get_context("fork")
        executor_kwargs = {"max_workers": workers, "mp_context": mp_ctx}
    else:
        executor_kwargs = {"max_workers": workers}

    all_results = {}

    for hp_overrides, hp_tag in combos:

        print("\n=====================================")
        print(f"[INFO] hyperparam combo: {hp_tag}")
        print("=====================================")

        if not parallel:
            res = {}
            for leave in letters:
                summary = _run_one_fold_search(cfg, table, letters, leave, hp_overrides, hp_tag)
                res[summary["tag"]] = summary
            all_results[hp_tag] = res
            continue

        with ProcessPoolExecutor(**executor_kwargs, initializer=_init_worker) as ex:
            futures = []
            for leave in letters:
                futures.append(
                    (leave, ex.submit(_run_one_fold_search, cfg, table, letters, leave, hp_overrides, hp_tag))
                )

            res = {}
            for leave, fut in futures:
                summary = fut.result()
                res[summary["tag"]] = summary

        all_results[hp_tag] = res

    print("=====================================")
    print("[INFO] tenfold hyperparameter search finished.")
    print("=====================================")
    return all_results