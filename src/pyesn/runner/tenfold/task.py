import cv2
from pathlib import Path

from . import utils
from .setup import setup_worker_seed

from pyesn.pipeline.trainer import Trainer
from pyesn.utils.data_processing import make_onehot
from pyesn.model.model_builder import get_model, get_model_param_str


def run_one_fold_search(
    cfg,
    csv_map: dict[str, Path],
    all_letters: list[str],
    leave_out_letter: str,
    hp_overrides: dict,
    weight_dir: Path,
    seed: int
):
    setup_worker_seed(seed)

    train_letters = [x for x in all_letters if x != leave_out_letter]
    train_tag = "".join(train_letters)
    print(f"[INFO] Start 10-fold train (leave_out='{leave_out_letter}') for hyperparams: {hp_overrides or 'default'}")
    
    csv_paths = [csv_map[ch] for ch in train_letters]
    ids, paths, class_ids = utils.read_data_from_csvs(csv_paths)
    assert len(ids) == len(paths) == len(class_ids), "length mismatch"

    trainer = Trainer(cfg.run_dir)
    model, optimizer = get_model(cfg, hp_overrides)

    Ny = cfg.model.Ny
    for i in range(len(paths)):
        img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {paths[i]}")
        U = img.T
        T = len(U)
        D = make_onehot(class_ids[i], T, Ny)
        trainer.train(model, optimizer, ids[i], U, D)

    weight_file_name = f"{get_model_param_str(cfg=cfg, overrides=hp_overrides)}_{train_tag}_Wout.npy"
    trainer.save_output_weight(
        Wout=model.Output.Wout,
        file_name=weight_file_name,
        save_dir=str(weight_dir)
    )
    print(f"[INFO] Finished fold '{leave_out_letter}'. Weight saved to {weight_dir / weight_file_name}")