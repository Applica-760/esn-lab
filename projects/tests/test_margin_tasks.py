from types import SimpleNamespace

import numpy as np
import pytest

from projects.tasks.analysis.margin.common import collect_samples, margin_rows
from projects.tasks.analysis.margin.distribution.app import (
    histogram_bin_edges,
)
from projects.tasks.analysis.margin.distribution.app import (
    main as analyze_margin,
)
from projects.tasks.pred_margin.app import validate_fold_correspondence
from projects.utils.prediction import load_pred_results, save_pred_results


def _label(class_index: int) -> np.ndarray:
    label = np.zeros((3, 3), dtype=float)
    label[:, class_index] = 1.0
    return label


def test_validate_fold_correspondence_requires_matching_known_ids():
    source_labels = [[_label(1), _label(0)] for _ in range(10)]
    source_ids = [["foraging", "other"] for _ in range(10)]
    known_labels = [[np.array([[1.0, 0.0]])] for _ in range(10)]
    known_ids = [["foraging"] for _ in range(10)]

    validate_fold_correspondence(source_labels, source_ids, known_labels, known_ids)

    known_ids[0] = ["different"]
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_fold_correspondence(source_labels, source_ids, known_labels, known_ids)


def test_histogram_bin_edges_use_shared_range():
    np.testing.assert_allclose(histogram_bin_edges((0.0, 3.3), 20), np.linspace(0.0, 3.3, 21))


def test_margin_analysis_writes_three_group_artifacts(tmp_path):
    result_base = tmp_path / "predictions" / "a" / "Nx_7" / "test_results"
    results = [
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
    ]
    save_pred_results(results, str(result_base))

    loaded = load_pred_results(str(result_base))
    assert loaded[0]["fold_index"] == 0
    assert [sample["id"] for sample in loaded[0]["results"]] == ["foraging", "rumination", "other"]
    assert loaded[0]["results"][0]["predictions"].shape == (3, 2)
    assert loaded[0]["results"][0]["labels"].shape == (3, 3)

    samples_by_param = collect_samples(tmp_path / "predictions", ["a"], warmup_ratio=1 / 3)
    rows_by_param = {name: margin_rows(samples) for name, samples in samples_by_param.items()}
    assert {row["true_label"] for row in rows_by_param["Nx_7"]} == {
        "foraging",
        "rumination",
        "other",
    }
    assert not collect_samples(
        tmp_path / "predictions", ["a"], warmup_ratio=1 / 3, fold_indices=[1]
    )

    output_dir = tmp_path / "analysis"
    analyze_margin(
        SimpleNamespace(
            pred_result_dir=tmp_path / "predictions",
            output_dir=output_dir,
            groups=["a"],
            warmup_ratio=1 / 3,
            bins=3,
            x_range=[0.0, 3.3],
        )
    )
    artifact_dir = output_dir / "Nx_7"
    assert (artifact_dir / "margin_by_sample.csv").exists()
    assert (artifact_dir / "margin_summary.csv").exists()
    assert (artifact_dir / "margin_distribution.png").exists()
