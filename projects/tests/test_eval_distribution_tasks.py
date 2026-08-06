from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from projects.tasks.eval.dist import app as dist_app
from projects.tasks.eval.dist_node import app as dist_node_app
from projects.utils.eval.judgment import save_judgment_results
from projects.utils.eval.sample_join import load_filtered_prediction_samples
from projects.utils.prediction import save_pred_results
from projects.utils.weights import build_param_str

PARAMS = {"Nx": 7, "input_scale": 0.001, "density": 0.5, "rho": 0.9}
PARAM_NAME = build_param_str(PARAMS)


def _sample(sample_id: str, true_label: int, pred_label: int) -> dict:
    labels = np.zeros((3, 2), dtype=float)
    labels[:, true_label] = 1.0
    predictions = np.full((3, 2), 0.1, dtype=float)
    predictions[:, pred_label] = 0.9
    return {"id": sample_id, "predictions": predictions, "labels": labels}


def _judgment(sample_id: str, true_label: int, pred_label: int) -> dict:
    return {
        "group": "a",
        "fold_index": 0,
        "id": sample_id,
        "pred_label": pred_label,
        "true_label": true_label,
        "is_correct": pred_label == true_label,
    }


def _write_inputs(tmp_path, judgments=None, samples=None):
    judge_dir = tmp_path / "judgments"
    pred_result_dir = tmp_path / "predictions"
    judgments = judgments or [
        _judgment("correct", true_label=0, pred_label=0),
        _judgment("incorrect", true_label=1, pred_label=0),
    ]
    samples = samples or [
        _sample("correct", true_label=0, pred_label=0),
        _sample("incorrect", true_label=1, pred_label=0),
    ]

    save_judgment_results(
        judgments,
        judge_dir / PARAM_NAME / "judgment_results_test.csv",
    )
    save_pred_results(
        [{"fold_index": 0, "results": samples}],
        str(pred_result_dir / "a" / PARAM_NAME / "test_results"),
    )
    return judge_dir, pred_result_dir


def test_load_filtered_prediction_samples_joins_filtered_results(tmp_path):
    judge_dir, pred_result_dir = _write_inputs(tmp_path)

    results = load_filtered_prediction_samples(
        PARAM_NAME,
        "test",
        judge_dir,
        pred_result_dir,
        {"true_label": 1},
    )

    assert len(results) == 1
    assert results[0]["id"] == "incorrect"
    assert results[0]["group"] == "a"
    assert results[0]["fold_index"] == 0
    np.testing.assert_array_equal(results[0]["labels"][:, 1], np.ones(3))


def test_load_filtered_prediction_samples_rejects_missing_key(tmp_path):
    judge_dir, pred_result_dir = _write_inputs(
        tmp_path,
        judgments=[_judgment("missing", true_label=0, pred_label=0)],
        samples=[_sample("different", true_label=0, pred_label=0)],
    )

    with pytest.raises(
        ValueError,
        match=r"missing prediction key: .*group='a'.*fold_index=0.*id='missing'",
    ):
        load_filtered_prediction_samples(PARAM_NAME, "test", judge_dir, pred_result_dir, {})


def test_load_filtered_prediction_samples_rejects_duplicate_judgment_key(tmp_path):
    judgment = _judgment("duplicate", true_label=0, pred_label=0)
    judge_dir, pred_result_dir = _write_inputs(
        tmp_path,
        judgments=[judgment, judgment],
        samples=[_sample("duplicate", true_label=0, pred_label=0)],
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate judgment key: .*group='a'.*fold_index=0.*id='duplicate'",
    ):
        load_filtered_prediction_samples(PARAM_NAME, "test", judge_dir, pred_result_dir, {})


def test_load_filtered_prediction_samples_rejects_duplicate_prediction_key(tmp_path):
    sample = _sample("duplicate", true_label=0, pred_label=0)
    judge_dir, pred_result_dir = _write_inputs(
        tmp_path,
        judgments=[_judgment("duplicate", true_label=0, pred_label=0)],
        samples=[sample, sample],
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate prediction key: .*group='a'.*fold_index=0.*id='duplicate'",
    ):
        load_filtered_prediction_samples(PARAM_NAME, "test", judge_dir, pred_result_dir, {})


class _ImmediateExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def _grid_config(**values):
    return SimpleNamespace(
        Nx=[PARAMS["Nx"]],
        input_scale=[PARAMS["input_scale"]],
        density=[PARAMS["density"]],
        rho=[PARAMS["rho"]],
        **values,
    )


def test_eval_distribution_tasks_keep_artifact_names(tmp_path, monkeypatch):
    judge_dir, pred_result_dir = _write_inputs(tmp_path)
    class_names = ["zero", "one"]
    class_order = [0, 1]
    colors = {
        "all": "#000000",
        "correct": "#111111",
        "incorrect": "#222222",
        "zero": "#333333",
        "one": "#444444",
    }

    monkeypatch.setattr(dist_app, "ProcessPoolExecutor", _ImmediateExecutor)
    dist_output_dir = tmp_path / "dist"
    dist_app.main(
        _grid_config(
            judge_dir=judge_dir,
            pred_result_dir=pred_result_dir,
            output_dir=dist_output_dir,
            filters={},
            mode=["test"],
            workers=1,
            bins=4,
            show_count=True,
            show_cumulative=False,
            colors=colors,
            class_names=class_names,
            class_order=class_order,
        )
    )

    assert (dist_output_dir / "intermediate" / f"{PARAM_NAME}_test_ratios.csv").exists()
    assert (dist_output_dir / "test" / "dist_confusion_test.png").exists()
    assert (dist_output_dir / "test" / "individual" / "dist_truezero_argmaxzero.png").exists()
    assert (dist_output_dir / "test" / "split" / "dist_split_trueone_test.png").exists()

    node_output_dir = tmp_path / "dist_node"
    dist_node_app.main(
        _grid_config(
            judge_dir=judge_dir,
            pred_result_dir=pred_result_dir,
            output_dir=node_output_dir,
            filters={},
            mode=["test"],
            bins=4,
            show_count=True,
            show_cumulative=False,
            colors=colors,
            class_names=class_names,
            class_order=class_order,
            metrics={
                "confidence": {"enabled": True, "xlabel": "confidence", "range": [0, 1]},
                "margin": {"enabled": True, "xlabel": "margin", "range": [0, 1]},
                "true_class_output": {
                    "enabled": True,
                    "xlabel": "true class output",
                    "range": [0, 1],
                },
                "node_output": {"enabled": True, "xlabel": "node output", "range": [0, 1]},
            },
        )
    )

    assert (node_output_dir / "test" / "node_confidence_all.png").exists()
    assert (node_output_dir / "test" / "node_margin_incorrect.png").exists()
    assert (node_output_dir / "test" / "node_true_class_output_one.png").exists()
    assert (node_output_dir / "test" / "node_output_confusion_test.png").exists()
