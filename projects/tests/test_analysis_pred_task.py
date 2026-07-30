from types import SimpleNamespace

import numpy as np

from projects.tasks.analysis.pred.app import main as analyze_predictions
from projects.utils.prediction import save_pred_results


def _sample(true_label: int) -> dict:
    labels = np.zeros((4, 2), dtype=float)
    labels[:, true_label] = 1.0
    predictions = np.array(
        [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], dtype=float
    )
    if true_label == 1:
        predictions = predictions[:, ::-1]
    return {"id": f"class-{true_label}", "predictions": predictions, "labels": labels}


def _fold(fold_index: int) -> dict:
    return {"fold_index": fold_index, "results": [_sample(0), _sample(1)]}


def test_analysis_pred_separates_modes_and_filters_folds(tmp_path):
    param_name = "Nx400_dens0.5_inscl0.001_rho0.9"
    for mode in ("train", "test"):
        save_pred_results(
            [_fold(0), _fold(1)],
            str(tmp_path / "predictions" / "b" / param_name / f"{mode}_results"),
        )

    output_dir = tmp_path / "analysis"
    analyze_predictions(
        SimpleNamespace(
            pred_result_dir=tmp_path / "predictions",
            output_dir=output_dir,
            groups=["b"],
            modes=["train", "test"],
            fold_indices=[0],
            warmup_ratio=0.0,
            n_bins=4,
            Nx=[400],
            input_scale=[0.001],
            density=[0.5],
            rho=[0.9],
            class_names=["foraging", "rumination"],
            class_order=[0, 1],
        )
    )

    for mode in ("train", "test"):
        fold_dir = output_dir / mode / param_name / "b" / "fold_0"
        assert (fold_dir / "score_trajectory_3x3.png").exists()
        assert not (output_dir / mode / param_name / "b" / "fold_1").exists()
