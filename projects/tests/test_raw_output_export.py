import csv
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from projects.tasks import cli
from projects.tasks.analysis.raw_output_export import app, plots
from projects.tasks.analysis.raw_output_export.data import collect_group
from projects.tasks.analysis.raw_output_viewer.data import ResultStore
from projects.tasks.analysis.raw_output_viewer.judgments import JudgmentStore, summarize_votes
from projects.utils.app_init import setup_task_environment
from projects.utils.eval.judgment import save_judgment_results
from projects.utils.prediction import save_pred_results


def config(tmp_path):
    return SimpleNamespace(
        pred_result_dir=tmp_path / "pred",
        judgment_result_dir=tmp_path / "judge",
        source_image_dir=tmp_path / "images",
        output_dir=tmp_path / "output",
        groups=["a", "b"],
        parameters=["p"],
        sample_ids=["cow"],
    )


def sample(sample_id="cow", values=None, label=0):
    values = np.asarray(values if values is not None else [[0.8, 0.2]] * 4)
    return dict(
        id=sample_id, predictions=values, labels=np.tile(np.eye(2)[label], (len(values), 1))
    )


def save(cfg, group, mode, folds):
    save_pred_results(
        [dict(fold_index=index, results=samples) for index, samples in folds],
        str(cfg.pred_result_dir / group / "p" / f"{mode}_results"),
    )


def test_export_separates_modes_groups_and_counts_actual_evaluations(tmp_path):
    cfg = config(tmp_path)
    save(cfg, "a", "test", [(0, [sample(values=[[0.1, 0.9]] * 4)]), (1, [sample()])])
    save(cfg, "a", "train", [(1, [sample(values=[[1, 1], [0, 1], [1, 0], [0, 1]])])])
    save(cfg, "b", "test", [(3, [sample(), sample("excluded")])])
    cfg.source_image_dir.mkdir()
    Image.new("L", (4, 2), 128).save(cfg.source_image_dir / "cow_ffmpeg.jpg")
    save_judgment_results(
        [dict(group="a", fold_index=0, id="cow", true_label=0, pred_label=0, is_correct=True)],
        cfg.judgment_result_dir / "p" / "judgment_results_test",
    )
    inputs = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    records = collect_group(
        ResultStore(cfg.pred_result_dir),
        {mode: JudgmentStore(cfg.judgment_result_dir) for mode in ("train", "test")},
        "a",
        "p",
        cfg.sample_ids,
        {},
    )["cow"]
    assert [(r["fold_index"], r["mode"]) for r in records] == [
        (0, "test"),
        (1, "train"),
        (1, "test"),
    ]
    app.main(cfg)
    paths = sorted(cfg.output_dir.rglob("*.png"))
    assert [str(p.relative_to(cfg.output_dir)) for p in paths] == ["p/cow/a.png", "p/cow/b.png"]
    for path in paths:
        with Image.open(path) as image:
            assert image.width == 1800
            image.verify()
    with (cfg.output_dir / "id_summary.csv").open() as source:
        rows = {r["mode"]: r for r in csv.DictReader(source)}
    assert (rows["test"]["evaluations"], rows["test"]["errors"]) == ("3", "1")
    assert float(rows["test"]["error_rate"]) == pytest.approx(1 / 3)
    assert rows["test"]["judgment_mismatches"] == "1"
    assert rows["test"]["judgment_unavailable"] == "2"
    # 時刻内・最終同票は採食。4票の同票を誤判定として数えない。
    assert (rows["train"]["evaluations"], rows["train"]["errors"]) == ("1", "0")
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}


@pytest.mark.parametrize("problem", ["length", "label"])
def test_id_identity_checked_within_and_across_groups(tmp_path, problem):
    cfg = config(tmp_path)
    save(cfg, "a", "train", [(0, [sample()])])
    altered = sample(values=[[0.8, 0.2]] * 3) if problem == "length" else sample(label=1)
    save(cfg, "b", "test", [(1, [altered])])
    store = ResultStore(cfg.pred_result_dir)
    judgments = {mode: JudgmentStore() for mode in ("train", "test")}
    identities = {}
    collect_group(store, judgments, "a", "p", [], identities)
    with pytest.raises(ValueError, match="inconsistent label or length"):
        collect_group(store, judgments, "b", "p", [], identities)
    save(cfg, "a", "test", [(1, [altered])])
    with pytest.raises(ValueError, match="inconsistent label or length"):
        collect_group(ResultStore(cfg.pred_result_dir), judgments, "a", "p", [], {})


@pytest.mark.parametrize("source_status", ["available", "not_found", "width_mismatch"])
def test_plot_preserves_values_axes_and_warning_states(monkeypatch, source_status):
    figures = []
    original = plots.FigureCanvasAgg.print_png

    def capture(self, output):
        result = original(self, output)
        figures.append(self.figure)
        return result

    monkeypatch.setattr(plots.FigureCanvasAgg, "print_png", capture)
    values = np.array([[3.0, -2.0], [0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    values.setflags(write=False)
    pixels = np.arange(8, dtype=np.uint8).reshape(2, 4)
    pixels.setflags(write=False)
    judgment = dict(status="available", pred_label="rumination")
    records = []
    for fold, mode, saved in [(0, "test", judgment), (1, "train", dict(status="not_found"))]:
        votes = summarize_votes(values, saved)
        records.append(
            dict(
                length=4,
                true_label="foraging",
                fold_index=fold,
                mode=mode,
                predictions=values,
                judgment=saved,
                votes=votes,
                is_correct=True,
                out_of_range=True,
            )
        )
    result = plots.render_group_png(
        "cow",
        "a",
        "p",
        records,
        pixels if source_status == "available" else None,
        dict(status=source_status),
    )
    assert result.startswith(b"\x89PNG\r\n\x1a\n")
    figure = figures[0]
    assert len(figure.axes) == 3
    image_axis, *output_axes = figure.axes
    if source_status == "available":
        np.testing.assert_array_equal(image_axis.images[0].get_array(), pixels)
        assert image_axis.images[0].get_extent() == [-0.5, 3.5, 1.5, -0.5]
    else:
        assert source_status in image_axis.texts[0].get_text()
    assert "Fold 0 | TEST" in output_axes[0].get_title()
    assert "MISMATCH (pred=rumination)" in output_axes[0].get_title()
    assert "vote=75.0%" in output_axes[0].get_title()
    assert "saved: not_found" in output_axes[1].get_title()
    for axis in output_axes:
        assert axis.get_ylim() == (-1.5, 2.5)
        assert "OUTPUT OUTSIDE" in axis.get_title()
        assert axis.get_xlim() == image_axis.get_xlim() == (-0.5, 3.5)
        for x in range(4):
            output_fraction = axis.transAxes.inverted().transform(axis.transData.transform((x, 0)))[
                0
            ]
            image_fraction = image_axis.transAxes.inverted().transform(
                image_axis.transData.transform((x, 0))
            )[0]
            assert output_fraction == pytest.approx(image_fraction)
        for index, line in enumerate(axis.lines):
            np.testing.assert_array_equal(line.get_ydata(), values[:, index])
            np.testing.assert_array_equal(line.get_xdata(), np.arange(4))
        assert [line.get_color() for line in axis.lines] == ["tab:blue", "tab:orange"]


@pytest.mark.parametrize("count", [1, 3, 4, 10])
def test_output_panels_use_three_columns(monkeypatch, count):
    figures = []

    def capture(self, output):
        self.draw()
        figures.append(self.figure)

    monkeypatch.setattr(plots.FigureCanvasAgg, "print_png", capture)
    values = np.array([[0.8, 0.2]] * 4)
    judgment = dict(status="disabled")
    record = dict(
        length=4,
        true_label="foraging",
        mode="train",
        predictions=values,
        judgment=judgment,
        votes=summarize_votes(values, judgment),
        is_correct=True,
        out_of_range=False,
    )
    plots.render_group_png(
        "cow",
        "a",
        "p",
        [{**record, "fold_index": index} for index in range(count)],
        np.zeros((2, 4), dtype=np.uint8),
        dict(status="available"),
    )
    image_axis, *axes = figures[0].axes
    assert len(axes) == count
    assert list(image_axis.get_subplotspec().colspan) == [0, 1, 2]
    for index, axis in enumerate(axes):
        spec = axis.get_subplotspec()
        assert spec.rowspan.start == 1 + index // 3
        assert spec.colspan.start == index % 3
        assert f"Fold {index} | TRAIN" in axis.get_title()


@pytest.mark.parametrize("name", ["../outside", "a/b", "a\\b", "..", "\x00"])
def test_unsafe_output_ids_are_rejected(tmp_path, name):
    cfg = config(tmp_path)
    save(cfg, "a", "test", [(0, [sample(name)])])
    cfg.sample_ids = []
    with pytest.raises(ValueError, match="ID"):
        app.main(cfg)
    assert not cfg.output_dir.exists()


def test_output_overlap_and_external_symlink_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg.output_dir = cfg.pred_result_dir
    with pytest.raises(ValueError, match="overlap"):
        app.main(cfg)
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "p").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="output path"):
        app._output_path(root, "p", "cow", "a.png")
    assert list(outside.iterdir()) == []


def test_cli_initialization_keeps_config_responsibility(tmp_path, monkeypatch):
    module = cli.TASK_REGISTRY["analysis.raw_output_export"]
    assert module == app.__name__
    task_dir = tmp_path / "projects/tasks/analysis/raw_output_export"
    task_dir.mkdir(parents=True)
    config_text = "output_dir: output\ngroups: [a]\nparameters: [p]\nsample_ids: []\n"
    (task_dir / "cfg.yaml").write_text(config_text)
    monkeypatch.chdir(tmp_path)
    cfg = setup_task_environment(module)
    assert str(cfg.output_dir) == "output"
    assert (tmp_path / "output/config.lock.yaml").read_text() == config_text
