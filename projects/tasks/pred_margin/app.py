from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from esn_lab.model.esn import ESN
from esn_lab.runner.pred.tenfold import pred_tenfold
from projects.utils.app_init import tenfold_data_loader
from projects.utils.prediction import is_valid_result_file, save_pred_results
from projects.utils.weights import list_param_dirs, load_metadata, load_tenfold_weights


def _sample_class(label: np.ndarray, expected_width: int, context: str) -> int:
    label = np.asarray(label)
    if label.ndim != 2 or label.shape[1] != expected_width:
        raise ValueError(
            f"{context}: expected label shape (timesteps, {expected_width}), got {label.shape}"
        )

    frame_classes = np.argmax(label, axis=1)
    if not np.all(frame_classes == frame_classes[0]):
        raise ValueError(f"{context}: labels must be single-class over all timesteps")
    return int(frame_classes[0])


def validate_fold_correspondence(source_labels, source_ids, known_labels, known_ids) -> None:
    """Verify that source foraging/rumination IDs match the 2-class training data per fold."""
    if not (len(source_labels) == len(source_ids) == len(known_labels) == len(known_ids) == 10):
        raise ValueError("source and known datasets must each contain exactly 10 folds")

    for fold_index, (src_labels, src_ids, two_labels, two_ids) in enumerate(
        zip(source_labels, source_ids, known_labels, known_ids)
    ):
        if len(src_labels) != len(src_ids) or len(two_labels) != len(two_ids):
            raise ValueError(f"fold {fold_index}: labels and IDs have different lengths")

        if len(set(src_ids)) != len(src_ids):
            raise ValueError(f"fold {fold_index}: duplicate IDs in source dataset")

        source_known_ids = set()
        source_other_ids = set()
        for label, sample_id in zip(src_labels, src_ids):
            true_class = _sample_class(label, 3, f"fold {fold_index}, source ID {sample_id}")
            if true_class == 0:
                source_other_ids.add(sample_id)
            else:
                source_known_ids.add(sample_id)

        known_dataset_ids = set()
        for label, sample_id in zip(two_labels, two_ids):
            _sample_class(label, 2, f"fold {fold_index}, 2-class ID {sample_id}")
            known_dataset_ids.add(sample_id)

        if len(known_dataset_ids) != len(two_ids):
            raise ValueError(f"fold {fold_index}: duplicate IDs in 2-class dataset")
        if source_known_ids != known_dataset_ids:
            missing = sorted(known_dataset_ids - source_known_ids)
            extra = sorted(source_known_ids - known_dataset_ids)
            raise ValueError(
                f"fold {fold_index}: source/2-class known ID mismatch "
                f"(missing_in_source={missing[:5]}, extra_in_source={extra[:5]})"
            )
        if source_other_ids & source_known_ids:
            raise ValueError(f"fold {fold_index}: IDs overlap between other and known samples")


def one_process(param_dir, group, source_data, source_labels, source_ids, cfg):
    param_str = param_dir.name
    result_path = Path(cfg.output_dir) / group / param_str / "test_results"
    if is_valid_result_file(str(result_path)):
        print(f"skipped (already exists): {param_str} group={group}")
        return

    params = load_metadata(param_dir)
    model = ESN(
        cfg.Nu, cfg.Ny, params["Nx"], params["density"], params["input_scale"], params["rho"]
    )
    results = pred_tenfold(
        model,
        load_tenfold_weights(param_dir),
        source_data,
        source_labels,
        source_ids,
        mode="test",
    )
    save_pred_results(results, str(result_path))
    print(f"proceed: {param_str} group={group}")


def main(cfg):
    weight_dir = Path(cfg.weight_dir)

    for group in cfg.groups:
        source_data, source_labels, source_ids = tenfold_data_loader(
            Path(cfg.data_source_base_dir) / group
        )
        known_labels, known_ids = tenfold_data_loader(
            Path(cfg.known_data_source_base_dir) / group
        )[1:]
        validate_fold_correspondence(source_labels, source_ids, known_labels, known_ids)

        param_dirs = list_param_dirs(weight_dir / group)
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = [
                executor.submit(
                    one_process,
                    param_dir,
                    group,
                    source_data,
                    source_labels,
                    source_ids,
                    cfg,
                )
                for param_dir in param_dirs
            ]
            for future in futures:
                future.result()

    print("margin prediction is finished")
