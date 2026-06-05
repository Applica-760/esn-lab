from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from projects.utils.prediction import load_pred_results, is_valid_result_file
from projects.utils.app_init import build_param_grid
from projects.utils.weights import build_param_str


# ── helpers ───────────────────────────────────────────────────────────────────

def strip_warmup(predictions, labels, warmup_ratio):
    T = len(predictions)
    warmup = int(T * warmup_ratio)
    return predictions[warmup:], labels[warmup:]


def get_bin_indices(T, n_bins):
    return np.minimum((np.arange(T) / T * n_bins).astype(int), n_bins - 1)


def collect_fold_samples(fold_data, warmup_ratio):
    """fold_data から warmup 除外済みサンプルリストを返す。true_label は labels の多数決、pred_label は argmax の多数決"""
    samples = []
    for sample in fold_data["results"]:
        preds, labels = strip_warmup(sample["predictions"], sample["labels"], warmup_ratio)
        true_frames = np.argmax(labels, axis=1)
        true_label = int(np.argmax(np.bincount(true_frames)))
        pred_frames = np.argmax(preds, axis=1)
        pred_label = int(np.argmax(np.bincount(pred_frames, minlength=preds.shape[1])))
        samples.append({"predictions": preds, "labels": labels, "true_label": true_label, "pred_label": pred_label})
    return samples


def _class_display_name(pred_class_idx, class_names, class_order):
    """予測クラスインデックス → 表示名"""
    return class_names[class_order.index(pred_class_idx)]


# ── temporal accuracy ─────────────────────────────────────────────────────────

def compute_temporal_accuracy(samples, n_bins):
    """Returns ndarray [n_classes, n_bins]: 相対位置ごとのフレーム正答率"""
    n_classes = samples[0]["predictions"].shape[1]
    correct_counts = np.zeros((n_classes, n_bins))
    total_counts = np.zeros((n_classes, n_bins))

    for s in samples:
        preds, labels = s["predictions"], s["labels"]
        T = len(preds)
        true_label = s["true_label"]
        frame_correct = (np.argmax(preds, axis=1) == np.argmax(labels, axis=1)).astype(float)
        bin_idx = get_bin_indices(T, n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                correct_counts[true_label, b] += frame_correct[mask].sum()
                total_counts[true_label, b] += mask.sum()

    denom = np.where(total_counts > 0, total_counts, np.nan)
    return correct_counts / denom


def plot_temporal_accuracy(acc, n_bins, class_names, class_order, output_dir):
    x = np.linspace(0, 100, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, class_idx in enumerate(class_order):
        ax.plot(x, acc[class_idx], label=class_names[i])
    ax.set_xlabel("Relative position (%)")
    ax.set_ylabel("Frame accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Temporal accuracy by true class")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "temporal_accuracy.png", dpi=150)
    plt.close(fig)


def analyze_temporal_accuracy(samples, n_bins, class_names, class_order, output_dir):
    acc = compute_temporal_accuracy(samples, n_bins)
    plot_temporal_accuracy(acc, n_bins, class_names, class_order, output_dir)
    return acc


# ── score trajectory ──────────────────────────────────────────────────────────

def analyze_score_trajectory(samples, n_bins, class_names, class_order, output_dir):
    """スコア軌跡 + std シェーディング（true class別）"""
    n_classes = len(class_names)
    # score_data[true_label][score_class][bin] = list of per-sample bin means
    score_data = [[[[] for _ in range(n_bins)] for _ in range(n_classes)] for _ in range(n_classes)]

    for s in samples:
        preds = s["predictions"]
        T = len(preds)
        true_label = s["true_label"]
        bin_idx = get_bin_indices(T, n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                bin_means = preds[mask].mean(axis=0)
                for sc in range(n_classes):
                    score_data[true_label][sc][b].append(bin_means[sc])

    x = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for i, class_idx in enumerate(class_order):
        ax = axes[i]
        ax.set_title(f"true = {class_names[i]}")
        for j, score_idx in enumerate(class_order):
            data_per_bin = score_data[class_idx][score_idx]
            means = np.array([np.mean(d) if d else np.nan for d in data_per_bin])
            stds = np.array([np.std(d) if len(d) > 1 else 0.0 for d in data_per_bin])
            color = f"C{j}"
            ax.plot(x, means, label=class_names[j], color=color)
            ax.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_xlabel("Relative position (%)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean score")
    fig.suptitle("Score trajectory by true class")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "score_trajectory.png", dpi=150)
    plt.close(fig)


# ── stability ─────────────────────────────────────────────────────────────────

def compute_stability(samples):
    """Returns list[list[float]]: サンプルごとの argmax 変動率（true class別）"""
    n_classes = samples[0]["predictions"].shape[1]
    instabilities = [[] for _ in range(n_classes)]
    for s in samples:
        preds = s["predictions"]
        true_label = s["true_label"]
        argmax_seq = np.argmax(preds, axis=1)
        change_rate = float(np.mean(argmax_seq[1:] != argmax_seq[:-1])) if len(argmax_seq) > 1 else 0.0
        instabilities[true_label].append(change_rate)
    return instabilities


def plot_stability(instabilities, class_names, class_order, output_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
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


def analyze_stability(samples, class_names, class_order, output_dir):
    instabilities = compute_stability(samples)
    plot_stability(instabilities, class_names, class_order, output_dir)
    return instabilities


# ── argmax heatmap ────────────────────────────────────────────────────────────

def analyze_argmax_heatmap(samples, n_bins, class_names, class_order, output_dir):
    """sample × 相対位置 → argmax クラスのヒートマップ（true class順ソート）"""
    n_classes = len(class_names)
    samples_sorted = sorted(samples, key=lambda s: s["true_label"])
    n_samples = len(samples_sorted)

    heatmap = np.full((n_samples, n_bins), fill_value=-1, dtype=int)
    for i, s in enumerate(samples_sorted):
        preds = s["predictions"]
        T = len(preds)
        bin_idx = get_bin_indices(T, n_bins)
        argmax_seq = np.argmax(preds, axis=1)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                heatmap[i, b] = np.argmax(np.bincount(argmax_seq[mask], minlength=n_classes))

    # 予測クラスインデックス j → 表示クラスと同じ色（C{rank}）
    colors = [f"C{class_order.index(j)}" for j in range(n_classes)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5), cmap.N)

    # true class 切り替わり位置に境界線を引く
    boundaries = [
        idx for idx in range(1, n_samples)
        if samples_sorted[idx]["true_label"] != samples_sorted[idx - 1]["true_label"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, n_samples // 8)))
    ax.imshow(heatmap, aspect="auto", cmap=cmap, norm=norm,
              extent=[0, 100, n_samples, 0], interpolation="nearest")
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=1.0)

    cbar = fig.colorbar(ax.images[0], ax=ax, ticks=range(n_classes))
    cbar.ax.set_yticklabels([_class_display_name(j, class_names, class_order) for j in range(n_classes)])
    ax.set_xlabel("Relative position (%)")
    ax.set_ylabel("Sample (sorted by true class)")
    ax.set_title("Argmax heatmap")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "argmax_heatmap.png", dpi=150)
    plt.close(fig)


# ── score margin ──────────────────────────────────────────────────────────────

def analyze_score_margin(samples, class_names, class_order, output_dir):
    """margin = max_score − 2nd_max_score のサンプル平均分布（true class別ヒストグラム）"""
    n_classes = len(class_names)
    margins = [[] for _ in range(n_classes)]

    for s in samples:
        preds = s["predictions"]
        true_label = s["true_label"]
        sorted_scores = np.sort(preds, axis=1)
        margin = float((sorted_scores[:, -1] - sorted_scores[:, -2]).mean())
        margins[true_label].append(margin)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for i, class_idx in enumerate(class_order):
        ax = axes[i]
        data = margins[class_idx]
        ax.hist(data, bins=20, edgecolor="white")
        ax.set_title(f"true = {class_names[i]} (n={len(data)})")
        ax.set_xlabel("Mean margin (max − 2nd max)")
        ax.set_ylabel("Count")
    fig.suptitle("Score margin distribution by true class")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "score_margin.png", dpi=150)
    plt.close(fig)


# ── 3×3 plots (true class × pred class) ──────────────────────────────────────

def _cell_samples(samples, true_idx, pred_idx):
    return [s for s in samples if s["true_label"] == true_idx and s["pred_label"] == pred_idx]


def plot_temporal_accuracy_3x3(samples, n_bins, class_names, class_order, output_dir):
    """フレーム正答率 3×3（row=true class, col=pred class）"""
    n_classes = len(class_names)
    x = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True)

    for row_i, true_idx in enumerate(class_order):
        for col_j, pred_idx in enumerate(class_order):
            ax = axes[row_i, col_j]
            cell = _cell_samples(samples, true_idx, pred_idx)
            ax.set_title(f"true={class_names[row_i]}, pred={class_names[col_j]} (n={len(cell)})", fontsize=8)
            if col_j == 0:
                ax.set_ylabel("Frame accuracy")
            if row_i == n_classes - 1:
                ax.set_xlabel("Relative position (%)", fontsize=7)
            ax.set_ylim(0, 1)
            if not cell:
                continue

            acc_per_bin = [[] for _ in range(n_bins)]
            for s in cell:
                preds, labels = s["predictions"], s["labels"]
                frame_correct = (np.argmax(preds, axis=1) == np.argmax(labels, axis=1)).astype(float)
                bin_idx = get_bin_indices(len(preds), n_bins)
                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.any():
                        acc_per_bin[b].append(frame_correct[mask].mean())

            means = np.array([np.mean(d) if d else np.nan for d in acc_per_bin])
            stds = np.array([np.std(d) if len(d) > 1 else 0.0 for d in acc_per_bin])
            ax.plot(x, means, color="steelblue")
            ax.fill_between(x, means - stds, means + stds, alpha=0.2, color="steelblue")

    fig.suptitle("Temporal accuracy (true × pred class)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "temporal_accuracy_3x3.png", dpi=150)
    plt.close(fig)


def plot_score_trajectory_3x3(samples, n_bins, class_names, class_order, output_dir):
    """スコア軌跡 3×3（row=true class, col=pred class）"""
    n_classes = len(class_names)
    x = np.linspace(0, 100, n_bins)
    fig, axes = plt.subplots(n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True)

    for row_i, true_idx in enumerate(class_order):
        for col_j, pred_idx in enumerate(class_order):
            ax = axes[row_i, col_j]
            cell = _cell_samples(samples, true_idx, pred_idx)
            ax.set_title(f"true={class_names[row_i]}, pred={class_names[col_j]} (n={len(cell)})", fontsize=8)
            if col_j == 0:
                ax.set_ylabel("Mean score")
            if row_i == n_classes - 1:
                ax.set_xlabel("Relative position (%)", fontsize=7)
            if not cell:
                continue

            score_data = [[[] for _ in range(n_bins)] for _ in range(n_classes)]
            for s in cell:
                preds = s["predictions"]
                bin_idx = get_bin_indices(len(preds), n_bins)
                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.any():
                        bin_means = preds[mask].mean(axis=0)
                        for sc in range(n_classes):
                            score_data[sc][b].append(bin_means[sc])

            for j, score_idx in enumerate(class_order):
                data_per_bin = score_data[score_idx]
                means = np.array([np.mean(d) if d else np.nan for d in data_per_bin])
                stds = np.array([np.std(d) if len(d) > 1 else 0.0 for d in data_per_bin])
                color = f"C{j}"
                ax.plot(x, means, label=class_names[j], color=color)
                ax.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)
            ax.legend(fontsize=6)

    fig.suptitle("Score trajectory (true × pred class)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "score_trajectory_3x3.png", dpi=150)
    plt.close(fig)


def plot_stability_3x3(samples, class_names, class_order, output_dir):
    """argmax変動率ヒストグラム 3×3（row=true class, col=pred class）"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True)

    for row_i, true_idx in enumerate(class_order):
        for col_j, pred_idx in enumerate(class_order):
            ax = axes[row_i, col_j]
            cell = _cell_samples(samples, true_idx, pred_idx)
            ax.set_title(f"true={class_names[row_i]}, pred={class_names[col_j]} (n={len(cell)})", fontsize=8)
            if col_j == 0:
                ax.set_ylabel("Count")
            if row_i == n_classes - 1:
                ax.set_xlabel("Argmax change rate", fontsize=7)
            if not cell:
                continue

            data = []
            for s in cell:
                argmax_seq = np.argmax(s["predictions"], axis=1)
                rate = float(np.mean(argmax_seq[1:] != argmax_seq[:-1])) if len(argmax_seq) > 1 else 0.0
                data.append(rate)
            ax.hist(data, bins=20, range=(0, 1), edgecolor="white")

    fig.suptitle("Within-sample stability (true × pred class)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stability_3x3.png", dpi=150)
    plt.close(fig)


def plot_score_margin_3x3(samples, class_names, class_order, output_dir):
    """score margin ヒストグラム 3×3（row=true class, col=pred class）"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_classes, figsize=(5 * n_classes, 4 * n_classes), sharey=True)

    for row_i, true_idx in enumerate(class_order):
        for col_j, pred_idx in enumerate(class_order):
            ax = axes[row_i, col_j]
            cell = _cell_samples(samples, true_idx, pred_idx)
            ax.set_title(f"true={class_names[row_i]}, pred={class_names[col_j]} (n={len(cell)})", fontsize=8)
            if col_j == 0:
                ax.set_ylabel("Count")
            if row_i == n_classes - 1:
                ax.set_xlabel("Mean margin (max − 2nd max)", fontsize=7)
            if not cell:
                continue

            margins = []
            for s in cell:
                preds = s["predictions"]
                sorted_scores = np.sort(preds, axis=1)
                margins.append(float((sorted_scores[:, -1] - sorted_scores[:, -2]).mean()))
            ax.hist(margins, bins=20, edgecolor="white")

    fig.suptitle("Score margin distribution (true × pred class)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "score_margin_3x3.png", dpi=150)
    plt.close(fig)


# ── fold summary ──────────────────────────────────────────────────────────────

def summarize_temporal_accuracy(fold_accs, n_bins, class_names, class_order, output_dir):
    """fold別 accuracy 曲線を重ね描き + mean"""
    x = np.linspace(0, 100, n_bins)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for i, class_idx in enumerate(class_order):
        ax = axes[i]
        ax.set_title(f"true = {class_names[i]}")
        fold_curves = np.array([acc[class_idx] for acc in fold_accs])
        for curve in fold_curves:
            ax.plot(x, curve, alpha=0.3, linewidth=0.8, color="steelblue")
        ax.plot(x, np.nanmean(fold_curves, axis=0), linewidth=2.0, color="steelblue", label="mean")
        ax.set_xlabel("Relative position (%)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frame accuracy")
    fig.suptitle("Temporal accuracy across folds")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "temporal_accuracy_folds.png", dpi=150)
    plt.close(fig)


def summarize_stability(fold_indices, fold_stabs, class_names, class_order, output_dir):
    """fold別 argmax 変動率分布を violin plot で表示"""
    n_classes = len(class_names)
    n_folds = len(fold_indices)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharey=True)
    for i, class_idx in enumerate(class_order):
        ax = axes[i]
        ax.set_title(f"true = {class_names[i]}")
        positions, plot_data = [], []
        for f, _ in enumerate(fold_indices):
            d = fold_stabs[f][class_idx]
            if len(d) > 1:
                positions.append(f + 1)
                plot_data.append(d)
        if plot_data:
            ax.violinplot(plot_data, positions=positions, showmedians=True)
        ax.set_xticks(range(1, n_folds + 1))
        ax.set_xticklabels([str(fi) for fi in fold_indices], fontsize=7)
        ax.set_xlabel("Fold")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Argmax change rate")
    fig.suptitle("Stability distribution across folds")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stability_folds.png", dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

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
            fold_results = []

            for fold_data in pred_results:
                fold_index = fold_data["fold_index"]
                samples = collect_fold_samples(fold_data, cfg.warmup_ratio)
                if not samples:
                    continue

                output_dir = cfg.output_dir / param_str / group / f"fold_{fold_index}"
                acc = analyze_temporal_accuracy(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                analyze_score_trajectory(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                stab = analyze_stability(samples, cfg.class_names, cfg.class_order, output_dir)
                analyze_argmax_heatmap(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                analyze_score_margin(samples, cfg.class_names, cfg.class_order, output_dir)
                plot_temporal_accuracy_3x3(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                plot_score_trajectory_3x3(samples, cfg.n_bins, cfg.class_names, cfg.class_order, output_dir)
                plot_stability_3x3(samples, cfg.class_names, cfg.class_order, output_dir)
                plot_score_margin_3x3(samples, cfg.class_names, cfg.class_order, output_dir)
                fold_results.append((fold_index, acc, stab))
                print(f"done: {param_str} group={group} fold={fold_index}")

            if fold_results:
                fold_indices = [r[0] for r in fold_results]
                fold_accs = [r[1] for r in fold_results]
                fold_stabs = [r[2] for r in fold_results]
                summary_dir = cfg.output_dir / param_str / group / "summary"
                summarize_temporal_accuracy(fold_accs, cfg.n_bins, cfg.class_names, cfg.class_order, summary_dir)
                summarize_stability(fold_indices, fold_stabs, cfg.class_names, cfg.class_order, summary_dir)
                print(f"summary: {param_str} group={group}")

    print("analysis finished")
