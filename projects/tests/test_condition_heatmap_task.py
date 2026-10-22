import csv
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from projects.tasks.analysis.condition_heatmap.app import collect_rows, main
from projects.tasks.analysis.condition_heatmap.plots import (
    _combined_cells,
    build_heatmap,
    render_heatmaps,
)
from projects.utils.eval.judgment import save_judgment_results
from projects.utils.prediction import save_pred_results


def config(tmp_path):
    return SimpleNamespace(
        pred_result_dir=tmp_path / "pred",
        judgment_result_dir=tmp_path / "judge",
        output_dir=tmp_path / "output",
        groups=["a", "b"],
        parameters=["p", "q"],
        fold_indices=[0, 1],
        modes=["train", "test"],
    )


def save(
    cfg,
    predictions,
    *,
    sample_id="cow",
    label=1,
    parameter="p",
    group="a",
    fold=0,
    mode="test",
    duplicate=False,
):
    sample = dict(
        id=sample_id,
        predictions=np.asarray(predictions, dtype=float),
        labels=np.tile(np.eye(2)[label], (len(predictions), 1)),
    )
    path = cfg.pred_result_dir / group / parameter / f"{mode}_results"
    save_pred_results(
        [dict(fold_index=fold, results=[sample] * (2 if duplicate else 1))], str(path)
    )
    return path.with_suffix(".npz")


def saved(cfg, *, sample_id="cow", label=1, pred=0, correct=False):
    save_judgment_results(
        [
            dict(
                group="a",
                fold_index=0,
                id=sample_id,
                true_label=label,
                pred_label=pred,
                is_correct=correct,
            )
        ],
        cfg.judgment_result_dir / "p" / "judgment_results_test",
    )


def cell(rows, *, parameter="p", group="a", fold=0, mode="test", sample_id="cow"):
    return next(
        r
        for r in rows
        if (r["parameter"], r["group"], int(r["fold_index"]), r["mode"], r["id"])
        == (parameter, group, fold, mode, sample_id)
    )


def test_votes_conditions_missing_and_csv_preserve_inputs(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[1, 1], [0, 1]])  # 時刻同値と最終同票は採食、真値は反芻。
    save(cfg, [[0, 1], [0, 1]], mode="train", fold=1)
    save(cfg, [[0, 1], [0, 1]], parameter="q")
    saved(cfg)
    paths = list(cfg.pred_result_dir.rglob("*.npz")) + list(cfg.judgment_result_dir.rglob("*.csv"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = collect_rows(cfg)
    assert len(rows) == 16
    row = cell(rows)
    assert row["true_ratio"] == 0.5
    assert row["foraging_votes"] == row["rumination_votes"] == 1
    assert row["is_correct"] is False
    assert row["matches_saved_judgment"] is True
    assert cell(rows, mode="train", fold=1)["true_ratio"] == 1
    assert cell(rows, parameter="q")["judgment_status"] == "not_found"
    missing = cell(rows, fold=1)
    assert missing["status"] == "missing_sample" and "true_ratio" not in missing
    assert cell(rows, group="b")["status"] == "missing_file"
    main(cfg)
    with (cfg.output_dir / "condition_cells.csv").open() as source:
        exported = list(csv.DictReader(source))
    assert len(exported) == len(rows)
    assert cell(exported, group="b")["true_ratio"] == ""
    for parameter in cfg.parameters:
        for mode in [*cfg.modes, "combined"]:
            assert (
                (cfg.output_dir / parameter / f"{mode}_001.png")
                .read_bytes()
                .startswith(b"\x89PNG\r\n\x1a\n")
            )
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@pytest.mark.parametrize("predictions,ratio", [([[1, 0], [1, 0]], 0), ([[0, 1]], 1)])
def test_zero_and_all_votes(tmp_path, predictions, ratio):
    cfg = config(tmp_path)
    save(cfg, predictions)
    assert cell(collect_rows(cfg))["true_ratio"] == ratio


def test_saved_mismatch_and_invalid_judgment(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[0, 1]])
    saved(cfg)
    assert cell(collect_rows(cfg))["status"] == "judgment_mismatch"
    saved(cfg, label=0, correct=True)
    row = cell(collect_rows(cfg))
    assert row["status"] == "invalid_judgment"
    assert row["judgment_status"] == "invalid_data"


def test_saved_only_id_is_retained(tmp_path):
    cfg = config(tmp_path)
    saved(cfg)
    row = cell(collect_rows(cfg))
    assert row["status"] == "missing_file"
    assert row["saved_is_correct"] is False
    assert "true_ratio" not in row


@pytest.mark.parametrize("change", ["nan", "label", "metadata"])
def test_invalid_input_is_not_zero(tmp_path, change):
    cfg = config(tmp_path)
    path = save(cfg, [[1, 0]])
    with np.load(path) as archive:
        data = dict(archive)
    if change == "nan":
        data["predictions"][0, 0] = np.nan
    elif change == "label":
        data["labels"][0] = [1, 1]
    else:
        data["ids"] = np.array(["cow", "extra"])
    np.savez(path, **data)
    saved(cfg)
    row = cell(collect_rows(cfg))
    assert row["status"] == ("invalid_file" if change == "metadata" else "invalid_data")
    assert "true_ratio" not in row
    assert row["true_label"] == "rumination"
    assert row["judgment_status"] == "available"


def test_multiple_ids_keep_missing_cells_and_invalid_saved_state(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[1, 0]], sample_id="first", label=0)
    save(cfg, [[0, 1]], sample_id="second", group="b")
    saved(cfg, sample_id="first", label=0, correct=True)
    path = cfg.judgment_result_dir / "p" / "judgment_results_test.csv"
    path.write_text(path.read_text().replace("True", "unknown"))
    rows = collect_rows(cfg)
    assert len(rows) == 32
    first = cell(rows, sample_id="first")
    assert first["status"] == "invalid_judgment"
    assert first["judgment_status"] == "invalid_data"
    assert cell(rows, sample_id="second")["status"] == "missing_sample"
    assert cell(rows, sample_id="second", group="b")["is_correct"] is True


@pytest.mark.parametrize("problem", ["duplicate", "length", "label"])
def test_conflicting_ids_fail_before_writing(tmp_path, problem):
    cfg = config(tmp_path)
    save(cfg, [[1, 0]], duplicate=problem == "duplicate")
    if problem != "duplicate":
        save(
            cfg,
            [[1, 0]] * (2 if problem == "length" else 1),
            group="b",
            label=0 if problem == "label" else 1,
        )
    with pytest.raises(ValueError, match="duplicate|inconsistent"):
        main(cfg)
    assert not cfg.output_dir.exists()


def test_rejects_overlapping_output_and_empty_inputs(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="no usable IDs"):
        main(cfg)
    cfg.output_dir = cfg.pred_result_dir / "output"
    with pytest.raises(ValueError, match="overlap"):
        main(cfg)


def test_heatmap_keeps_zero_missing_and_invalid_distinct(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[1, 0]])
    saved(cfg)
    rows = collect_rows(cfg)
    cell(rows, group="b", fold=1)["status"] = "judgment_mismatch"
    cells = {
        (r["id"], r["group"], r["fold_index"]): r
        for r in rows
        if r["parameter"] == "p" and r["mode"] == "test"
    }
    figure = build_heatmap(
        ["cow"],
        [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
        cells,
        {"cow": "rumination"},
        "p | test",
    )
    axis = figure.axes[0]
    image = axis.images[0]
    assert image.get_clim() == (0, 100)
    assert image.get_array()[0, 0] == 0
    np.testing.assert_array_equal(
        np.ma.getmaskarray(image.get_array()), [[False, True, True, True]]
    )
    assert {(t.get_position(), t.get_text()) for t in axis.texts} == {
        ((0, 0), "X"),
        ((2, 0), "-"),
        ((3, 0), "!"),
    }
    assert [t.get_text() for t in axis.get_xticklabels()] == ["a/0", "a/1", "b/0", "b/1"]
    assert axis.get_yticklabels()[0].get_text() == "cow [rumination]"
    assert [line.get_xdata()[0] for line in axis.lines] == [0.5, 1.5, 2.5]
    assert axis.lines[1].get_linewidth() > axis.lines[0].get_linewidth()
    assert len(axis.lines[1].get_path_effects()) == 2
    incorrect = next(t for t in axis.texts if t.get_text() == "X")
    assert incorrect.get_color() == "white"
    assert incorrect.get_fontweight() == "bold"


def test_pages_share_id_order_across_modes_and_parameters(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    cfg = config(tmp_path)
    save(cfg, [[1, 0]], sample_id="z", label=0)
    save(cfg, [[0, 1]], sample_id="a", group="b")
    rows = collect_rows(cfg)
    pages = []

    def capture(figure, path, **kwargs):
        axis = figure.axes[0]
        pages.append((path, [t.get_text() for t in axis.get_yticklabels()]))

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = render_heatmaps(
        rows,
        cfg.output_dir,
        cfg.groups,
        cfg.fold_indices,
        cfg.parameters,
        cfg.modes,
        rows_per_page=1,
    )
    assert len(paths) == 12
    for index in range(0, 12, 2):
        assert pages[index][1] == ["z [foraging]"]
        assert pages[index + 1][1] == ["a [rumination]"]
        assert pages[index][0].name.endswith("_001.png")
        assert pages[index + 1][0].name.endswith("_002.png")


def test_cli_dispatches_heatmap_with_default_config(tmp_path, monkeypatch):
    import sys

    from projects.tasks import cli
    from projects.tasks.analysis.condition_heatmap import app
    from projects.utils.app_init import load_config

    cfg = load_config("projects/tasks/analysis/condition_heatmap/cfg.yaml")
    cfg.output_dir = tmp_path / "output"
    dispatched = []
    monkeypatch.setattr(sys, "argv", ["cli", "analysis.condition_heatmap"])
    monkeypatch.setattr(cli, "setup_task_environment", lambda *a, **kw: cfg)
    monkeypatch.setattr(app, "main", lambda value: dispatched.append(value))
    cli.main()
    assert dispatched == [cfg]
    assert cfg.groups == list("abcdefghij")
    assert cfg.fold_indices == list(range(10))


def test_plot_rejects_output_symlink(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[1, 0]])
    rows = collect_rows(cfg)
    cfg.output_dir.mkdir()
    (cfg.output_dir / "p").symlink_to(cfg.pred_result_dir / "a" / "p", target_is_directory=True)
    with pytest.raises(ValueError, match="inside output"):
        render_heatmaps(
            rows, cfg.output_dir, cfg.groups, cfg.fold_indices, cfg.parameters, cfg.modes
        )


def test_combined_uses_both_modes_and_outlines_only_test(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[1, 0]])
    save(cfg, [[0, 1]], mode="train", fold=1)
    saved(cfg)
    rows = [r for r in collect_rows(cfg) if r["parameter"] == "p"]
    cells = _combined_cells(rows)
    figure = build_heatmap(
        ["cow"],
        [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
        cells,
        {"cow": "rumination"},
        "combined",
        combined=True,
    )
    axis = figure.axes[0]
    np.testing.assert_allclose(axis.images[0].get_array()[0, :2], [0, 100])
    assert len(axis.patches) == 1
    border = axis.patches[0]
    assert border.get_xy() == (-0.44, -0.44)
    assert border.get_edgecolor() == (1, 1, 1, 1)
    assert len(border.get_path_effects()) == 2
    assert {t.get_text() for t in axis.get_legend().get_texts()} == {
        "No usable prediction",
        "Test: black/white outline",
        "Train: no outline",
    }
    assert cells["cow", "b", 0]["status"] == "missing_file"
    cell(rows, mode="train")["status"] = "invalid_file"
    assert _combined_cells(rows)["cow", "a", 0]["status"] == "invalid_file"


def test_combined_rejects_overlapping_modes(tmp_path):
    cfg = config(tmp_path)
    save(cfg, [[1, 0]])
    save(cfg, [[0, 1]], mode="train")
    with pytest.raises(ValueError, match="train/test overlap"):
        render_heatmaps(
            collect_rows(cfg),
            cfg.output_dir,
            cfg.groups,
            cfg.fold_indices,
            cfg.parameters,
            cfg.modes,
        )
