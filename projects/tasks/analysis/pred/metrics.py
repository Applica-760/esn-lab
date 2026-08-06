import numpy as np


def bin_indices(length, n_bins):
    return np.minimum((np.arange(length) / length * n_bins).astype(int), n_bins - 1)


def compute_temporal_accuracy(samples, n_bins):
    """相対位置ごとのフレーム正答率をクラス別に返す。"""
    n_classes = samples[0]["predictions"].shape[1]
    correct_counts = np.zeros((n_classes, n_bins))
    total_counts = np.zeros((n_classes, n_bins))

    for sample in samples:
        predictions = sample["predictions"]
        labels = sample["labels"]
        frame_correct = (np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)).astype(float)
        indices = bin_indices(len(predictions), n_bins)
        for bin_index in range(n_bins):
            mask = indices == bin_index
            if mask.any():
                true_label = sample["true_label"]
                correct_counts[true_label, bin_index] += frame_correct[mask].sum()
                total_counts[true_label, bin_index] += mask.sum()

    denominator = np.where(total_counts > 0, total_counts, np.nan)
    return correct_counts / denominator


def compute_score_trajectory(samples, n_bins):
    """true class・score class・相対位置ごとの平均と標準偏差を返す。"""
    n_classes = samples[0]["predictions"].shape[1]
    values = [[[[] for _ in range(n_bins)] for _ in range(n_classes)] for _ in range(n_classes)]

    for sample in samples:
        predictions = sample["predictions"]
        indices = bin_indices(len(predictions), n_bins)
        for bin_index in range(n_bins):
            mask = indices == bin_index
            if mask.any():
                bin_means = predictions[mask].mean(axis=0)
                for score_class in range(n_classes):
                    values[sample["true_label"]][score_class][bin_index].append(
                        bin_means[score_class]
                    )

    means = np.full((n_classes, n_classes, n_bins), np.nan)
    standard_deviations = np.zeros((n_classes, n_classes, n_bins))
    for true_class in range(n_classes):
        for score_class in range(n_classes):
            for bin_index, bin_values in enumerate(values[true_class][score_class]):
                if bin_values:
                    means[true_class, score_class, bin_index] = np.mean(bin_values)
                    if len(bin_values) > 1:
                        standard_deviations[true_class, score_class, bin_index] = np.std(bin_values)
    return means, standard_deviations


def compute_stability(samples):
    """サンプルごとのargmax変動率をtrue class別に返す。"""
    n_classes = samples[0]["predictions"].shape[1]
    instabilities = [[] for _ in range(n_classes)]
    for sample in samples:
        argmax_sequence = np.argmax(sample["predictions"], axis=1)
        change_rate = (
            float(np.mean(argmax_sequence[1:] != argmax_sequence[:-1]))
            if len(argmax_sequence) > 1
            else 0.0
        )
        instabilities[sample["true_label"]].append(change_rate)
    return instabilities


def compute_argmax_heatmap(samples, n_bins):
    """true class順のargmaxヒートマップとクラス境界を返す。"""
    n_classes = samples[0]["predictions"].shape[1]
    sorted_samples = sorted(samples, key=lambda sample: sample["true_label"])
    heatmap = np.full((len(sorted_samples), n_bins), fill_value=-1, dtype=int)

    for sample_index, sample in enumerate(sorted_samples):
        predictions = sample["predictions"]
        indices = bin_indices(len(predictions), n_bins)
        argmax_sequence = np.argmax(predictions, axis=1)
        for bin_index in range(n_bins):
            mask = indices == bin_index
            if mask.any():
                heatmap[sample_index, bin_index] = np.argmax(
                    np.bincount(argmax_sequence[mask], minlength=n_classes)
                )

    boundaries = [
        index
        for index in range(1, len(sorted_samples))
        if sorted_samples[index]["true_label"] != sorted_samples[index - 1]["true_label"]
    ]
    return heatmap, boundaries


def compute_score_margins(samples):
    """サンプル平均score marginをtrue class別に返す。"""
    n_classes = samples[0]["predictions"].shape[1]
    margins = [[] for _ in range(n_classes)]
    for sample in samples:
        sorted_scores = np.sort(sample["predictions"], axis=1)
        margin = float((sorted_scores[:, -1] - sorted_scores[:, -2]).mean())
        margins[sample["true_label"]].append(margin)
    return margins


def _samples_by_class_pair(samples, class_order):
    return {
        (true_class, pred_class): [
            sample
            for sample in samples
            if sample["true_label"] == true_class and sample["pred_label"] == pred_class
        ]
        for true_class in class_order
        for pred_class in class_order
    }


def compute_cell_temporal_accuracy(samples, n_bins, class_order):
    """true×predセルごとのフレーム正答率の平均と標準偏差を返す。"""
    result = {}
    for key, cell_samples in _samples_by_class_pair(samples, class_order).items():
        values = [[] for _ in range(n_bins)]
        for sample in cell_samples:
            predictions = sample["predictions"]
            labels = sample["labels"]
            frame_correct = (np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)).astype(
                float
            )
            indices = bin_indices(len(predictions), n_bins)
            for bin_index in range(n_bins):
                mask = indices == bin_index
                if mask.any():
                    values[bin_index].append(frame_correct[mask].mean())

        result[key] = {
            "count": len(cell_samples),
            "means": np.array([np.mean(items) if items else np.nan for items in values]),
            "stds": np.array([np.std(items) if len(items) > 1 else 0.0 for items in values]),
        }
    return result


def compute_cell_score_trajectory(samples, n_bins, class_order):
    """true×predセルごとのscore軌跡の平均と標準偏差を返す。"""
    n_classes = samples[0]["predictions"].shape[1]
    result = {}
    for key, cell_samples in _samples_by_class_pair(samples, class_order).items():
        values = [[[] for _ in range(n_bins)] for _ in range(n_classes)]
        for sample in cell_samples:
            predictions = sample["predictions"]
            indices = bin_indices(len(predictions), n_bins)
            for bin_index in range(n_bins):
                mask = indices == bin_index
                if mask.any():
                    bin_means = predictions[mask].mean(axis=0)
                    for score_class in range(n_classes):
                        values[score_class][bin_index].append(bin_means[score_class])

        means = np.full((n_classes, n_bins), np.nan)
        standard_deviations = np.zeros((n_classes, n_bins))
        for score_class in range(n_classes):
            for bin_index, bin_values in enumerate(values[score_class]):
                if bin_values:
                    means[score_class, bin_index] = np.mean(bin_values)
                    if len(bin_values) > 1:
                        standard_deviations[score_class, bin_index] = np.std(bin_values)
        result[key] = {
            "count": len(cell_samples),
            "means": means,
            "stds": standard_deviations,
        }
    return result


def compute_cell_stability(samples, class_order):
    """true×predセルごとのargmax変動率を返す。"""
    result = {}
    for key, cell_samples in _samples_by_class_pair(samples, class_order).items():
        values = []
        for sample in cell_samples:
            argmax_sequence = np.argmax(sample["predictions"], axis=1)
            change_rate = (
                float(np.mean(argmax_sequence[1:] != argmax_sequence[:-1]))
                if len(argmax_sequence) > 1
                else 0.0
            )
            values.append(change_rate)
        result[key] = values
    return result


def compute_cell_score_margins(samples, class_order):
    """true×predセルごとのサンプル平均score marginを返す。"""
    result = {}
    for key, cell_samples in _samples_by_class_pair(samples, class_order).items():
        values = []
        for sample in cell_samples:
            sorted_scores = np.sort(sample["predictions"], axis=1)
            values.append(float((sorted_scores[:, -1] - sorted_scores[:, -2]).mean()))
        result[key] = values
    return result
