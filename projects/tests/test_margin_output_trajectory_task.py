from types import SimpleNamespace

import numpy as np

from projects.tasks.analysis.margin.output_trajectory.app import main as analyze_trajectory
from projects.utils.prediction import save_pred_results


def _label(class_index: int) -> np.ndarray:
    label = np.zeros((3, 3), dtype=float)
    label[:, class_index] = 1.0
    return label


def test_output_trajectory_writes_combined_and_fold_artifacts(tmp_path):
    result_base = tmp_path / "predictions" / "a" / "Nx_7" / "test_results"
    save_pred_results(
        [
            {
                "fold_index": 0,
                "results": [
                    {
                        "id": "foraging",
                        "predictions": np.array([[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]]),
                        "labels": _label(1),
                    },
                    {
                        "id": "rumination",
                        "predictions": np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 2.0]]),
                        "labels": _label(2),
                    },
                    {
                        "id": "other",
                        "predictions": np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
                        "labels": _label(0),
                    },
                ],
            }
        ],
        str(result_base),
    )

    output_dir = tmp_path / "trajectory"
    analyze_trajectory(
        SimpleNamespace(
            pred_result_dir=tmp_path / "predictions",
            output_dir=output_dir,
            groups=["a"],
            warmup_ratio=1 / 3,
            trajectory_bins=3,
            fold_indices=[0],
            separate_fold_output=True,
        )
    )

    artifact_dir = output_dir / "Nx_7"
    assert (artifact_dir / "score_trajectory_3x2.png").exists()
    assert (artifact_dir / "a" / "fold_0" / "score_trajectory_3x2.png").exists()
    assert not (artifact_dir / "margin_by_sample.csv").exists()
