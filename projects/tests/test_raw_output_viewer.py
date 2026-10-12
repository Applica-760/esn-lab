import base64
import json
import re
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
    assert store.list_sample_folds("a", PARAMETER, "same-id") == [
        {"fold": 0, "modes": ["train", "test"]}
    ]
    assert len(calls) == 2


@pytest.mark.parametrize("extension", ["npz", "json"])
def test_sample_folds_lists_membership_and_reuses_plot_selection(tmp_path, extension):
    for mode, indices in (("train", [3, 1, 2]), ("test", [3, 0])):
        path = write_results(tmp_path, mode=mode, extension=extension)
        folds = [
            {
                "fold_index": index,
                "results": [{**sample(), "id": "other-id" if index == 2 else "same-id"}],
            }
            for index in indices
        ]
        if extension == "json":
            path.write_text(json.dumps(folds))
        else:
            save_pred_results(folds, str(path))
    write_results(tmp_path, group="b", fold=8)
    api = client(tmp_path)
    query = {key: QUERY[key] for key in ("group", "parameter", "id")}
    response = api.get("/api/sample-folds", params=query)
    assert response.status_code == 200
    assert response.json() == {
        "folds": [
            {"fold": 0, "modes": ["test"]},
            {"fold": 1, "modes": ["train"]},
            {"fold": 2, "modes": []},
            {"fold": 3, "modes": ["train", "test"]},
        ]
    }
    selected = response.json()["folds"][1]
    plot = api.get(
        "/api/plots", params={**query, "fold": selected["fold"], "mode": selected["modes"][0]}
    )
    assert plot.status_code == 200
    assert (plot.json()["id"], plot.json()["fold"], plot.json()["mode"]) == ("same-id", 1, "train")
    assert api.get("/api/sample-folds", params={**query, "group": "b"}).json() == {
        "folds": [{"fold": 8, "modes": ["test"]}]
    }
    assert api.get("/api/sample-folds", params=query).json() == response.json()


def test_sample_folds_errors_and_json_precedence(tmp_path):
    write_results(tmp_path)
    path = write_results(tmp_path, samples=[{**sample(), "id": "json-id"}], extension="json")
    api = client(tmp_path)
    query = {"group": "a", "parameter": PARAMETER, "id": "json-id"}
    assert api.get("/api/sample-folds", params=query).status_code == 200
    for update in (
        {"id": "same-id"},
        {"id": "json"},
        {"group": "missing"},
        {"parameter": "missing"},
    ):
        response = api.get("/api/sample-folds", params={**query, **update})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
    for key in query:
        response = api.get("/api/sample-folds", params={k: v for k, v in query.items() if k != key})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
    response = api.get("/api/sample-folds", params={**query, "group": ".."})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_data"
    path.write_text("invalid json")
    response = client(tmp_path).get("/api/sample-folds", params=query)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_data"


@pytest.mark.parametrize(
    "value,warn", [(-1.6, True), (-1.5, False), (1.3, False), (2.5, False), (2.6, True)]
)
def test_api_returns_three_pngs_with_range_warning(tmp_path, value, warn):
    write_results(tmp_path, samples=[sample(value=value)])
    api = client(tmp_path)
    assert api.get("/api/conditions").json() == {"groups": ["a"]}
    assert api.get("/api/samples", params=QUERY).json()["samples"][0]["id"] == "same-id"
    response = api.get("/api/plots", params=QUERY)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "same-id" and body["length"] == 4
    assert body["mime_type"] == "image/png" and body["encoding"] == "base64"
    assert set(body["images"]) == {"foraging", "rumination", "overlay"}
    for encoded in body["images"].values():
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            assert image.format == "PNG"
            assert image.size == (1000, 300)
            image.verify()
    assert [w["node"] for w in body["warnings"]] == (["foraging"] if warn else [])
    if warn:
        assert body["warnings"][0]["code"] == "out_of_range"
        assert "−1.5〜2.5" in body["warnings"][0]["message"]


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
    assert axes.get_ylim() == (-1.5, 2.5)
    assert axes.get_xlim() == (0, 2)
    combined = np.column_stack((values, values[::-1]))
    viewer_plots.render_node_png(combined, "foraging / rumination")
    axes = captured[1]
    assert len(axes.lines) == 2
    for index, line in enumerate(axes.lines):
        np.testing.assert_array_equal(line.get_ydata(), combined[:, index])
        np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
    assert [line.get_color() for line in axes.lines] == ["tab:blue", "tab:orange"]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["foraging", "rumination"]
    assert axes.get_ylim() == (-1.5, 2.5)


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


def test_frontend_assets_are_served_without_exposing_source(tmp_path):
    api = client(tmp_path)
    response = api.get("/")
    assert response.status_code == 200
    assert 'id="samples"' in response.text
    assert response.headers["cache-control"] == "no-store"
    for asset in ("style.css", "app.js"):
        url = re.search(rf"/assets/{re.escape(asset)}\?v=\d+", response.text).group()
        assert api.get(url).status_code == 200
    assert 'id="foraging"' in response.text
    assert 'id="rumination"' in response.text
    assert 'id="overlay"' in response.text
    assert api.get("/assets/app.js").status_code == 200
    assert api.get("/assets/style.css").status_code == 200
    assert api.get("/assets/%2e%2e/%2e%2e/tasks/cli.py").status_code == 404
