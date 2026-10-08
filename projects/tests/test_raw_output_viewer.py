import base64
import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from projects.tasks import cli
from projects.tasks.analysis.raw_output_viewer import app as viewer_app
from projects.tasks.analysis.raw_output_viewer import data as viewer_data
from projects.tasks.analysis.raw_output_viewer import plots as viewer_plots
from projects.utils.prediction import save_pred_results

PARAMETER = "Nx400_dens0.5_inscl0.001_rho0.9"
QUERY = {"group": "a", "parameter": PARAMETER, "fold": 0, "mode": "test", "id": "same-id"}


def sample(label=0, length=4, value=0.8):
    return {
        "id": "same-id",
        "predictions": [[value, 0.2]] * length,
        "labels": [[1 - label, label]] * length,
    }


def write_results(root, mode="test", samples=None, extension="npz", group="a", fold=0):
    base = root / group / PARAMETER / f"{mode}_results"
    base.parent.mkdir(parents=True, exist_ok=True)
    results = [{"fold_index": fold, "results": samples if samples is not None else [sample()]}]
    if extension == "npz":
        save_pred_results(results, str(base))
    else:
        base.with_suffix(".json").write_text(json.dumps(results))
    return base.with_suffix(f".{extension}")


def client(root):
    return TestClient(viewer_app.create_app(SimpleNamespace(pred_result_dir=root)))


@pytest.mark.parametrize("extension", ["npz", "json"])
def test_saved_results_conditions_and_exact_sample_selection(tmp_path, extension):
    precise = 0.81234567890123
    write_results(tmp_path, samples=[sample(value=precise)], extension=extension)
    write_results(tmp_path, "train", [sample(label=1, length=7)], extension)
    write_results(tmp_path, group="b", fold=1, samples=[sample(value=0.3)])
    store = viewer_data.ResultStore(tmp_path)
    assert store.list_conditions() == {"groups": ["a", "b"]}
    assert store.list_conditions("a") == {"parameters": [PARAMETER]}
    assert store.list_conditions("a", PARAMETER) == {"folds": [0]}
    assert store.list_samples("a", PARAMETER, 0) == [
        {"id": "same-id", "mode": "test", "true_label": "foraging", "length": 4},
        {"id": "same-id", "mode": "train", "true_label": "rumination", "length": 7},
    ]
    assert store.get_sample("a", PARAMETER, 0, "test", "same-id")["predictions"][0, 0] == precise
    assert store.get_sample("a", PARAMETER, 0, "train", "same-id")["true_label"] == "rumination"
    assert store.get_sample("b", PARAMETER, 1, "test", "same-id")["predictions"][0, 0] == 0.3
    with pytest.raises(FileNotFoundError):
        store.get_sample("b", PARAMETER, 0, "test", "same-id")


def test_selected_train_test_files_are_not_reloaded(tmp_path, monkeypatch):
    write_results(tmp_path)
    write_results(tmp_path, "train")
    original = viewer_data.load_pred_results
    calls = []

    def counted(base):
        calls.append(base)
        return original(base)

    monkeypatch.setattr(viewer_data, "load_pred_results", counted)
    store = viewer_data.ResultStore(tmp_path)
    store.list_conditions("a", PARAMETER)
    store.list_samples("a", PARAMETER, 0)
    store.get_sample("a", PARAMETER, 0, "test", "same-id")
    store.get_sample("a", PARAMETER, 0, "train", "same-id")
    assert len(calls) == 2


def test_api_returns_two_pngs_with_range_warning(tmp_path):
    write_results(tmp_path, samples=[sample(value=1.3)])
    api = client(tmp_path)
    assert api.get("/api/conditions").json() == {"groups": ["a"]}
    assert api.get("/api/samples", params=QUERY).json()["samples"][0]["id"] == "same-id"
    response = api.get("/api/plots", params=QUERY)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "same-id" and body["length"] == 4
    assert body["mime_type"] == "image/png" and body["encoding"] == "base64"
    assert set(body["images"]) == {"foraging", "rumination"}
    for encoded in body["images"].values():
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            assert image.format == "PNG"
            image.verify()
    assert [w["node"] for w in body["warnings"]] == ["foraging"]
    assert body["warnings"][0]["code"] == "out_of_range"


def test_plot_keeps_original_values_and_time_axis(monkeypatch):
    captured = []
    canvas = viewer_plots.FigureCanvasAgg
    original = canvas.print_png

    def inspect(self, output):
        captured.append(self.figure.axes[0])
        return original(self, output)

    monkeypatch.setattr(canvas, "print_png", inspect)
    values = np.array([-0.5, 0.4, 1.5])
    viewer_plots.render_node_png(values, "foraging")
    axes = captured[0]
    np.testing.assert_array_equal(axes.lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_array_equal(axes.lines[0].get_ydata(), values)
    assert axes.get_ylim() == (-0.2, 1.2)
    assert axes.get_xlim() == (0, 2)


@pytest.mark.parametrize(
    "bad_sample",
    [
        {"id": "same-id", "predictions": [], "labels": []},
        {"id": "same-id", "predictions": [[0.2, 0.3, 0.5]], "labels": [[1, 0, 0]]},
        {"id": "same-id", "predictions": [[None, 0.3]], "labels": [[1, 0]]},
        {"id": "same-id", "predictions": [[float("nan"), 0.3]], "labels": [[1, 0]]},
        {"id": "same-id", "predictions": [[0.2, 0.3]], "labels": [[0, 0]]},
        {"id": "same-id", "predictions": [[0.2, 0.3]] * 2, "labels": [[1, 0], [0, 1]]},
    ],
)
def test_invalid_sample_is_reported(tmp_path, bad_sample):
    write_results(tmp_path, samples=[bad_sample], extension="json")
    response = client(tmp_path).get("/api/plots", params=QUERY)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_data"


def test_missing_invalid_requests_and_render_failure(tmp_path, monkeypatch):
    write_results(tmp_path)
    api = client(tmp_path)
    for update in ({"id": "missing"}, {"fold": 9}, {"mode": "train"}):
        response = api.get("/api/plots", params={**QUERY, **update})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
    assert api.get("/api/plots", params={**QUERY, "group": ".."}).status_code == 422
    response = api.get("/api/plots", params={**QUERY, "mode": "wrong"})
    assert response.json()["error"]["code"] == "invalid_request"

    def fail(*args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(viewer_app, "render_node_png", fail)
    response = api.get("/api/plots", params=QUERY)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "render_failed"


def test_corrupt_npz_length_and_duplicate_ids(tmp_path):
    path = write_results(tmp_path)
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["lengths"] = np.array([99])
    np.savez(path, **arrays)
    assert client(tmp_path).get("/api/plots", params=QUERY).status_code == 422
    write_results(tmp_path, samples=[sample(), sample()], extension="json")
    assert client(tmp_path).get("/api/plots", params=QUERY).status_code == 422


def test_cli_registration_and_local_start(monkeypatch):
    assert cli.TASK_REGISTRY["analysis.raw_output_viewer"] == viewer_app.__name__
    calls = []
    monkeypatch.setattr(viewer_app.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    viewer_app.main(SimpleNamespace(pred_result_dir="unused", port=8000))
    assert calls == [{"host": "127.0.0.1", "port": 8000}]
