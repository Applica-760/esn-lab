import numpy as np


def majority_vote_prediction(predictions: list) -> int:
    """
    フレーム単位の予測から多数決で1つのラベルを決定

    Args:
        predictions: フレームごとの予測値リスト [[p0, p1, p2], ...]
                     各フレームでargmaxを取り、最頻出クラスを返す

    Returns:
        多数決で決定されたクラスインデックス
    """
    predictions = np.array(predictions)
    frame_labels = np.argmax(predictions, axis=1)
    counts = np.bincount(frame_labels, minlength=predictions.shape[1])
    return int(np.argmax(counts))


def majority_vote_label(labels: list) -> int:
    """
    正解ラベルの集約（現在は多数決、将来差し替え可能）

    Args:
        labels: フレームごとの正解ラベルリスト [[1,0,0], ...]
                one-hot形式を想定

    Returns:
        多数決で決定されたクラスインデックス
    """
    labels = np.array(labels)
    frame_labels = np.argmax(labels, axis=1)
    counts = np.bincount(frame_labels, minlength=labels.shape[1])
    return int(np.argmax(counts))
