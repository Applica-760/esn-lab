import csv
from pathlib import Path

from projects.tasks.analysis.raw_output_export.data import (
    collect_group,
    summarize_ids,
    validate_names,
)
from projects.tasks.analysis.raw_output_export.plots import render_group_png
from projects.tasks.analysis.raw_output_viewer.data import ResultStore, load_source_image
from projects.tasks.analysis.raw_output_viewer.judgments import JudgmentStore


def _output_path(root, *parts):
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"output path outside directory or symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(cfg):
    validate_names(cfg.groups, "groups")
    validate_names(cfg.parameters, "parameters")
    validate_names(cfg.sample_ids, "sample_ids", allow_empty=True)
    output = Path(cfg.output_dir).resolve()
    for value in (cfg.pred_result_dir, cfg.judgment_result_dir, cfg.source_image_dir):
        if value:
            source = Path(value).resolve()
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("output directory must not overlap an input directory")
    store = ResultStore(cfg.pred_result_dir)
    judgments = {mode: JudgmentStore(cfg.judgment_result_dir) for mode in ("train", "test")}
    identities = {}
    rows = []
    count = 0
    for parameter in cfg.parameters:
        for group in cfg.groups:
            comparisons = collect_group(
                store, judgments, group, parameter, cfg.sample_ids, identities
            )
            for sample_id, records in sorted(comparisons.items()):
                pixels, info = load_source_image(
                    cfg.source_image_dir, sample_id, records[0]["length"]
                )
                image = render_group_png(sample_id, group, parameter, records, pixels, info)
                path = _output_path(output, parameter, sample_id, f"{group}.png")
                path.write_bytes(image)
                rows.extend(
                    dict(
                        parameter=parameter,
                        id=sample_id,
                        true_label=r["true_label"],
                        mode=r["mode"],
                        is_correct=r["is_correct"],
                        judgment_status=r["judgment"]["status"],
                        matches_saved_judgment=r["votes"]["matches_saved_judgment"],
                    )
                    for r in records
                )
                count += 1
                print(f"saved: {path}", flush=True)
                del records
            # 次のGroupの大容量結果を読む前に、前の系列への参照を解放する。
            del comparisons
    if not rows:
        raise ValueError("no samples matched the configured IDs")
    missing = set(cfg.sample_ids) - identities.keys()
    if missing:
        raise ValueError(f"configured IDs not found: {sorted(missing)}")
    summaries = summarize_ids(rows)
    with _output_path(output, "id_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"{count} PNGs, {len(summaries)} ID/mode summaries: {output}")
