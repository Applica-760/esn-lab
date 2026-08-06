from pathlib import Path
from types import SimpleNamespace

import numpy as np

from projects.tasks import cli as task_cli
from projects.tasks.analysis.pred.app import main as analyze_predictions
from projects.tasks.analysis.pred.metrics import (
    compute_argmax_heatmap,
    compute_cell_score_margins,
    compute_cell_score_trajectory,
    compute_cell_stability,
    compute_cell_temporal_accuracy,
    compute_score_margins,
    compute_score_trajectory,
    compute_stability,
    compute_temporal_accuracy,
)
from projects.tasks.analysis.pred.samples import normalize_fold_samples, strip_warmup
from projects.utils import app_init
from projects.utils.prediction import save_pred_results

FOLD_ARTIFACTS = {
    "temporal_accuracy.png",
    "score_trajectory.png",
    "stability.png",
    "argmax_heatmap.png",
    "score_margin.png",
    "temporal_accuracy_3x3.png",
    "score_trajectory_3x3.png",
    "stability_3x3.png",
    "score_margin_3x3.png",
}
SUMMARY_ARTIFACTS = {"temporal_accuracy_folds.png", "stability_folds.png"}


def _sample(true_label: int) -> dict:
    labels = np.zeros((4, 2), dtype=float)
    labels[:, true_label] = 1.0
    predictions = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], dtype=float)
    if true_label == 1:
        predictions = predictions[:, ::-1]
    return {"id": f"class-{true_label}", "predictions": predictions, "labels": labels}


def _fold(fold_index: int) -> dict:
    return {"fold_index": fold_index, "results": [_sample(0), _sample(1)]}


def _config(pred_result_dir, output_dir, **values):
    return SimpleNamespace(
        pred_result_dir=pred_result_dir,
        output_dir=output_dir,
        groups=["b"],
        warmup_ratio=0.0,
        n_bins=4,
        Nx=[400],
        input_scale=[0.001],
        density=[0.5],
        rho=[0.9],
        class_names=["foraging", "rumination"],
        class_order=[0, 1],
        **values,
    )


def test_analysis_pred_pure_sample_and_metric_calculations():
    fold_data = _fold(0)
    predictions, labels = strip_warmup(
        fold_data["results"][0]["predictions"],
        fold_data["results"][0]["labels"],
        0.25,
    )
    assert predictions.shape == labels.shape == (3, 2)

    samples = normalize_fold_samples(fold_data, warmup_ratio=0.0)
    assert [(sample["true_label"], sample["pred_label"]) for sample in samples] == [
        (0, 0),
        (1, 1),
    ]

    np.testing.assert_allclose(compute_temporal_accuracy(samples, 2), np.ones((2, 2)))
    trajectory_means, trajectory_stds = compute_score_trajectory(samples, 2)
    assert trajectory_means.shape == trajectory_stds.shape == (2, 2, 2)
    assert compute_stability(samples) == [[0.0], [0.0]]
    np.testing.assert_allclose(compute_score_margins(samples), [[0.5], [0.5]])

    heatmap, boundaries = compute_argmax_heatmap(samples, 2)
    np.testing.assert_array_equal(heatmap, np.array([[0, 0], [1, 1]]))
    assert boundaries == [1]

    temporal_cells = compute_cell_temporal_accuracy(samples, 2, [0, 1])
    trajectory_cells = compute_cell_score_trajectory(samples, 2, [0, 1])
    stability_cells = compute_cell_stability(samples, [0, 1])
    margin_cells = compute_cell_score_margins(samples, [0, 1])
    assert temporal_cells[(0, 0)]["count"] == 1
    assert trajectory_cells[(1, 1)]["count"] == 1
    assert stability_cells[(0, 1)] == []
    np.testing.assert_allclose(margin_cells[(1, 1)], [0.5])


def test_analysis_pred_separates_modes_and_filters_folds(tmp_path):
    param_name = "Nx400_dens0.5_inscl0.001_rho0.9"
    for mode in ("train", "test"):
        save_pred_results(
            [_fold(0), _fold(1)],
            str(tmp_path / "predictions" / "b" / param_name / f"{mode}_results"),
        )

    output_dir = tmp_path / "analysis"
    analyze_predictions(
        _config(
            tmp_path / "predictions",
            output_dir,
            modes=["train", "test"],
            fold_indices=[0],
        )
    )

    for mode in ("train", "test"):
        fold_dir = output_dir / mode / param_name / "b" / "fold_0"
        assert {path.name for path in fold_dir.iterdir()} == FOLD_ARTIFACTS
        summary_dir = output_dir / mode / param_name / "b" / "summary"
        assert {path.name for path in summary_dir.iterdir()} == SUMMARY_ARTIFACTS
        assert not (output_dir / mode / param_name / "b" / "fold_1").exists()
    assert not (output_dir / param_name).exists()


def test_analysis_pred_default_mode_keeps_legacy_output_hierarchy(tmp_path):
    param_name = "Nx400_dens0.5_inscl0.001_rho0.9"
    save_pred_results(
        [_fold(0), _fold(1)],
        str(tmp_path / "predictions" / "b" / param_name / "train_results"),
    )

    output_dir = tmp_path / "analysis"
    analyze_predictions(_config(tmp_path / "predictions", output_dir))

    for fold_index in (0, 1):
        fold_dir = output_dir / param_name / "b" / f"fold_{fold_index}"
        assert {path.name for path in fold_dir.iterdir()} == FOLD_ARTIFACTS
    assert {path.name for path in (output_dir / param_name / "b" / "summary").iterdir()} == (
        SUMMARY_ARTIFACTS
    )
    assert not (output_dir / "train").exists()


def test_analysis_pred_cli_and_config_resolution(tmp_path, monkeypatch):
    module_path = "projects.tasks.analysis.pred.app"
    assert task_cli.TASK_REGISTRY["analysis.pred"] == module_path

    loaded_paths = []
    copied_paths = []

    def fake_load_config(config_path):
        loaded_paths.append(Path(config_path))
        return SimpleNamespace(output_dir=tmp_path / f"output-{len(loaded_paths)}")

    monkeypatch.setattr(app_init, "load_config", fake_load_config)
    monkeypatch.setattr(
        app_init.shutil,
        "copy",
        lambda source, destination: copied_paths.append((Path(source), Path(destination))),
    )

    default_config = app_init.setup_task_environment(module_path)
    explicit_config = app_init.setup_task_environment(
        module_path, config_name="cfg_2class_trajectory.yaml"
    )

    task_dir = Path("projects/tasks/analysis/pred")
    assert loaded_paths == [task_dir / "cfg.yaml", task_dir / "cfg_2class_trajectory.yaml"]
    assert copied_paths == [
        (task_dir / "cfg.yaml", default_config.output_dir / "config.lock.yaml"),
        (
            task_dir / "cfg_2class_trajectory.yaml",
            explicit_config.output_dir / "config.lock.yaml",
        ),
    ]
