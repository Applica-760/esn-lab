from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from projects.utils.prediction import load_pred_results, is_valid_result_file
from projects.utils.app_init import build_param_grid
from projects.utils.weights import build_param_str


def strip_warmup(predictions, labels, warmup_ratio):
    T = len(predictions)
    warmup = int(T * warmup_ratio)
    return predictions[warmup:], labels[warmup:]


def collect_fold_samples(fold_data, warmup_ratio):
    """fold_data から warmup 除外済みサンプルリストを返す。true_label は labels の多数決 (ground truth のみ)"""
    samples = []
    for sample in fold_data["results"]:
        preds, labels = strip_warmup(sample["predictions"], sample["labels"], warmup_ratio)
        true_frames = np.argmax(labels, axis=1)
        true_label = int(np.argmax(np.bincount(true_frames)))
        samples.append({"predictions": preds, "labels": labels, "true_label": true_label})
    return samples


def analyze_temporal_accuracy(samples, n_bins, class_names, class_order, output_dir):
    """相対フレーム位置ごとのフレーム正答率を true class 別に集計してプロット"""
    # bin ごとの正解数・総数: [n_classes, n_bins]
    n_classes = len(class_names)
    correct_counts = np.zeros((n_classes, n_bins))
    total_counts = np.zeros((n_classes, n_bins))

    for s in samples:
        preds, labels = s["predictions"], s["labels"]
        T = len(preds)
        true_label = s["true_label"]
        frame_correct = (np.argmax(preds, axis=1) == np.argmax(labels, axis=1)).astype(float)
        bin_indices = np.minimum((np.arange(T) / T * n_bins).astype(int), n_bins - 1)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.any():
                correct_counts[true_label, b] += frame_correct[mask].sum()
                total_counts[true_label, b] += mask.sum()

    x = np.linspace(0, 100, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, class_idx in enumerate(class_order):
        name = class_names[i]
        denom = total_counts[class_idx]
        acc = np.where(denom > 0, correct_counts[class_idx] / denom, np.nan)
        ax.plot(x, acc, label=name)

    ax.set_xlabel("Relative position (%)")
    ax.set_ylabel("Frame accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Temporal accuracy by true class")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "temporal_accuracy.png", dpi=150)
    plt.close(fig)


def analyze_score_trajectory(samples, n_bins, class_names, class_order, output_dir):
    """相対フレーム位置ごとのクラス別スコア平均を true class 別にプロット"""
    n_classes = len(class_names)
    # score_sums / counts: [n_classes(true), n_classes(score), n_bins]
    score_sums = np.zeros((n_classes, n_classes, n_bins))
    score_counts = np.zeros((n_classes, n_bins))

    for s in samples:
        preds = s["predictions"]
        T = len(preds)
        true_label = s["true_label"]
        bin_indices = np.minimum((np.arange(T) / T * n_bins).astype(int), n_bins - 1)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.any():
                score_sums[true_label, :, b] += preds[mask].mean(axis=0)
                score_counts[true_label, b] += 1

    x = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for i, class_idx in enumerate(class_order):
        ax = axes[i]
        ax.set_title(f"true = {class_names[i]}")
        for j, score_idx in enumerate(class_order):
            denom = score_counts[class_idx]
            mean_score = np.where(denom > 0, score_sums[class_idx, score_idx] / denom, np.nan)
            ax.plot(x, mean_score, label=class_names[j])
        ax.set_xlabel("Relative position (%)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean score")
    fig.suptitle("Score trajectory by true class")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "score_trajectory.png", dpi=150)
    plt.close(fig)


def analyze_stability(samples, class_names, class_order, output_dir):
    """サンプル内 argmax 変動率の分布を true class 別にプロット"""
    n_classes = len(class_names)
    instabilities = [[] for _ in range(n_classes)]

    for s in samples:
        preds = s["predictions"]
        true_label = s["true_label"]
        argmax_seq = np.argmax(preds, axis=1)
        if len(argmax_seq) > 1:
            change_rate = np.mean(argmax_seq[1:] != argmax_seq[:-1])
        else:
            change_rate = 0.0
        instabilities[true_label].append(change_rate)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=False)
    for i, class_idx in enumerate(class_order):
        ax = axes[i]
        data = instabilities[class_idx]
        ax.hist(data, bins=20, range=(0, 1), edgecolor="white")
        ax.set_title(f"true = {class_names[i]} (n={len(data)})")
        ax.set_xlabel("Argmax change rate")
        ax.set_ylabel("Count")

    fig.suptitle("Within-sample stability by true class")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stability.png", dpi=150)
    plt.close(fig)


def main(cfg):
    pred_result_dir = Path(cfg.pred_result_dir)
    param_grid = build_param_grid(cfg)

    for params in param_grid:
        param_str = build_param_str(params)

        for group in cfg.groups:
            result_base = str(pred_result_dir / group / param_str / "train_results")
            if not is_valid_result_file(result_base):
                print(f"skip (not found): {param_str} group={group}")
                continue

            pred_results = load_pred_results(result_base)

            for fold_data in pred_results:
                fold_index = fold_data["fold_index"]
                samples = collect_fold_samples(fold_data, cfg.warmup_ratio)
                if not samples:
                    continue

                output_dir = cfg.output_dir / param_str / group / f"fold_{fold_index}"
                analyze_temporal_accuracy(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                analyze_score_trajectory(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                analyze_stability(samples, cfg.class_names, cfg.class_order, output_dir)
                print(f"done: {param_str} group={group} fold={fold_index}")

    print("analysis finished")
