from pathlib import Path

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
from projects.tasks.analysis.pred.plots import (
    plot_argmax_heatmap,
    plot_fold_stability,
    plot_fold_temporal_accuracy,
    plot_score_margin,
    plot_score_margin_cells,
    plot_score_trajectory,
    plot_score_trajectory_cells,
    plot_stability,
    plot_stability_cells,
    plot_temporal_accuracy,
    plot_temporal_accuracy_cells,
)
from projects.tasks.analysis.pred.samples import normalize_fold_samples
from projects.utils.app_init import build_param_grid
from projects.utils.prediction import is_valid_result_file, load_pred_results
from projects.utils.weights import build_param_str


def _analyze_fold(samples, cfg, output_dir):
    accuracy = compute_temporal_accuracy(samples, cfg.n_bins)
    plot_temporal_accuracy(accuracy, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)

    score_means, score_stds = compute_score_trajectory(samples, cfg.n_bins)
    plot_score_trajectory(
        score_means,
        score_stds,
        cfg.n_bins,
        cfg.class_names,
        cfg.class_order,
        output_dir,
    )

    stability = compute_stability(samples)
    plot_stability(stability, cfg.class_names, cfg.class_order, output_dir)

    heatmap, boundaries = compute_argmax_heatmap(samples, cfg.n_bins)
    plot_argmax_heatmap(heatmap, boundaries, cfg.class_names, cfg.class_order, output_dir)

    margins = compute_score_margins(samples)
    plot_score_margin(margins, cfg.class_names, cfg.class_order, output_dir)

    temporal_cells = compute_cell_temporal_accuracy(samples, cfg.n_bins, cfg.class_order)
    plot_temporal_accuracy_cells(
        temporal_cells,
        cfg.n_bins,
        cfg.class_names,
        cfg.class_order,
        output_dir,
    )

    trajectory_cells = compute_cell_score_trajectory(samples, cfg.n_bins, cfg.class_order)
    plot_score_trajectory_cells(
        trajectory_cells,
        cfg.n_bins,
        cfg.class_names,
        cfg.class_order,
        output_dir,
    )

    stability_cells = compute_cell_stability(samples, cfg.class_order)
    plot_stability_cells(stability_cells, cfg.class_names, cfg.class_order, output_dir)

    margin_cells = compute_cell_score_margins(samples, cfg.class_order)
    plot_score_margin_cells(margin_cells, cfg.class_names, cfg.class_order, output_dir)
    return accuracy, stability


def _output_root(cfg, mode, separate_mode_output):
    output_root = Path(cfg.output_dir)
    return output_root / mode if separate_mode_output else output_root


def main(cfg):
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)
    modes = getattr(cfg, "modes", ["train"])
    selected_fold_indices = set(getattr(cfg, "fold_indices", []))
    separate_mode_output = hasattr(cfg, "modes")

    for params in param_grid:
        param_name = build_param_str(params)
        for mode in modes:
            for group in cfg.groups:
                result_base = str(pred_result_dir / group / param_name / f"{mode}_results")
                if not is_valid_result_file(result_base):
                    print(f"skip (not found): {param_name} group={group} mode={mode}")
                    continue

                fold_results = []
                for fold_data in load_pred_results(result_base):
                    fold_index = fold_data["fold_index"]
                    if selected_fold_indices and fold_index not in selected_fold_indices:
                        continue

                    samples = normalize_fold_samples(fold_data, cfg.warmup_ratio)
                    if not samples:
                        continue

                    output_root = _output_root(cfg, mode, separate_mode_output)
                    output_dir = output_root / param_name / group / f"fold_{fold_index}"
                    accuracy, stability = _analyze_fold(samples, cfg, output_dir)
                    fold_results.append((fold_index, accuracy, stability))
                    print(f"done: {param_name} group={group} mode={mode} fold={fold_index}")

                if not fold_results:
                    continue

                fold_indices = [result[0] for result in fold_results]
                fold_accuracies = [result[1] for result in fold_results]
                fold_stabilities = [result[2] for result in fold_results]
                summary_dir = (
                    _output_root(cfg, mode, separate_mode_output) / param_name / group / "summary"
                )
                plot_fold_temporal_accuracy(
                    fold_accuracies,
                    cfg.n_bins,
                    cfg.class_names,
                    cfg.class_order,
                    summary_dir,
                )
                plot_fold_stability(
                    fold_indices,
                    fold_stabilities,
                    cfg.class_names,
                    cfg.class_order,
                    summary_dir,
                )
                print(f"summary: {param_name} group={group} mode={mode}")

    print("analysis finished")
