from pathlib import Path

from .setup import setup_worker_seed
from pyesn.pipeline.train.tenfold_trainer import TenfoldTrainer


def run_one_fold_search(
    cfg,
    csv_map: dict[str, Path],
    all_letters: list[str],
    leave_out_letter: str,
    hp_overrides: dict,
    weight_dir: Path,
    seed: int,
) -> tuple[float, str]:
    """Thin wrapper: set seed (runner responsibility) and delegate to pipeline TenfoldTrainer."""
    setup_worker_seed(seed)
    trainer = TenfoldTrainer(cfg.run_dir)
    return trainer.run_one_fold_search(
        cfg=cfg,
        csv_map=csv_map,
        all_letters=all_letters,
        leave_out_letter=leave_out_letter,
        hp_overrides=hp_overrides,
        weight_dir=weight_dir,
    )