# runner/train.py
import os
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from itertools import product
from concurrent.futures import ProcessPoolExecutor

from mypkg.utils.data_processing import make_onehot
from mypkg.pipeline.trainer import Trainer
from mypkg.model.model_builder import get_model_with_overrides, get_model_param_str


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

def _run_one_fold_search(cfg, table, letters, leave, hp_overrides: dict, hp_tag: str):
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

    trainer.save_output_weight(Wout=model.Output.Wout, 
                               filename=f"{get_model_param_str(cfg=cfg, overrides=hp_overrides)}_{tag}", 
                               save_dir="./artifacts/weights/")

    return {
        "hp_tag": hp_tag,
        "tag" : tag,
        "num_samples": len(paths),
        "letters_used": train_letters,
    }





def tenfold_search_train(cfg, *, parallel: bool = True, max_workers: int = 10):
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

        # このハイパーパラメータでの結果を格納する辞書
        fold_results = {}
        tasks_to_run = []

        # 1. 全てのfoldをチェックし、実行タスクとスキップタスクを振り分ける
        # for leave in letters:
        #     train_letters = [x for x in letters if x != leave]
        #     tag = "".join(train_letters)

        #     weight_filename = f"{hp_tag}_{tag}*"
        #     print(weight_filename)
            
        #     expected_path = f"./artifacts/{}/{weight_filename}"

        #     if expected_path.exists():
        #         print(f"[INFO] Weight file found, skipping fold '{leave}': {expected_path}")
        #         # スキップした結果を先に記録
        #         summary = {
        #             "hp_tag": hp_tag,
        #             "tag": tag,
        #             "num_samples": "skipped",
        #             "letters_used": train_letters,
        #             "status": "skipped",
        #         }
        #         fold_results[tag] = summary
        #     else:
        #         # 重みファイルが存在しない場合、実行リストに追加
        #         tasks_to_run.append(leave)

        # 2. 実行が必要なタスクがなければ次のハイパラへ
        if not tasks_to_run:
            print("[INFO] All folds for this combo are already trained.")
            all_results[hp_tag] = fold_results
            continue

        # 3. 実行が必要なタスクを実行
        if not parallel:
            for leave in tasks_to_run:
                summary = _run_one_fold_search(cfg, table, letters, leave, hp_overrides, hp_tag)
                fold_results[summary["tag"]] = summary
        else:
            with ProcessPoolExecutor(**executor_kwargs, initializer=_init_worker) as ex:
                futures = []
                for leave in tasks_to_run:
                    futures.append(
                        (leave, ex.submit(_run_one_fold_search, cfg, table, letters, leave, hp_overrides, hp_tag))
                    )

                for leave, fut in futures:
                    try:
                        summary = fut.result()
                        fold_results[summary["tag"]] = summary
                    except Exception as e:
                        print(f"[ERROR] Fold '{leave}' in combo '{hp_tag}' failed: {e}")
                        # エラーが発生した場合も記録を残す
                        train_letters = [x for x in letters if x != leave]
                        tag = "".join(train_letters)
                        fold_results[tag] = {"status": "failed", "error": str(e), "tag": tag, "hp_tag": hp_tag}
        
        all_results[hp_tag] = fold_results

    print("=====================================")
    print("[INFO] tenfold hyperparameter search finished.")
    print("=====================================")
    return all_results