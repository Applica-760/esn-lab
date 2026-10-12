import csv
from pathlib import Path

import numpy as np

from projects.utils.eval.judgment import load_judgment_results


class JudgmentStore:
    """現在のparameter・modeの保存判定を保持する。"""

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None
        self._selection = None
        self._rows = {}
        self._error = None

    def _load(self, parameter, mode):
        if self._selection == (parameter, mode):
            return
        self._selection = (parameter, mode)
        self._rows = {}
        self._error = None
        try:
            if (
                not parameter
                or parameter in (".", "..")
                or Path(parameter).name != parameter
                or mode not in ("train", "test")
            ):
                raise ValueError("判定結果の条件指定が不正です。")
            path = (self.root / parameter / f"judgment_results_{mode}.csv").resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("判定CSVは設定したディレクトリ内に置いてください。")
            # 共通ローダーはTrue以外をFalseに変換するため、元の表記も検証する。
            with path.open(newline="") as source:
                reader = csv.DictReader(source)
                required = {"group", "fold_index", "id", "pred_label", "true_label", "is_correct"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError("判定CSVの必須列がありません。")
                for row in reader:
                    if row["is_correct"] not in ("True", "False"):
                        raise ValueError("判定CSVの正誤表記が不正です。")
            rows = {}
            for row in load_judgment_results(path):
                key = (row["group"], row["fold_index"], row["id"])
                if not row["group"] or not row["id"] or row["fold_index"] < 0:
                    raise ValueError("判定CSVのGroup・Fold・IDが不正です。")
                if key in rows:
                    raise ValueError("判定CSVのGroup・Fold・IDが重複しています。")
                if row["pred_label"] not in (0, 1) or row["true_label"] not in (0, 1):
                    raise ValueError("判定CSVのラベルは0または1である必要があります。")
                if row["is_correct"] != (row["pred_label"] == row["true_label"]):
                    raise ValueError("判定CSVのラベルと正誤が一致しません。")
                rows[key] = row
            self._rows = rows
        except FileNotFoundError:
            self._error = {"status": "not_found", "message": "判定CSVが見つかりません。"}
        except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
            self._error = {"status": "invalid_data", "message": f"判定CSVを利用できません: {exc}"}

    def get_judgment(self, group, parameter, fold, mode, sample_id, true_label):
        if self.root is None:
            return {"status": "disabled", "message": "判定結果の参照先が未設定です。"}
        self._load(parameter, mode)
        if self._error:
            return dict(self._error)
        row = self._rows.get((group, fold, sample_id))
        if row is None:
            return {"status": "not_found", "message": "対応する判定結果がありません。"}
        labels = ("foraging", "rumination")
        if labels[row["true_label"]] != true_label:
            return {"status": "invalid_data", "message": "判定CSVと生出力の真値が一致しません。"}
        return {
            "status": "available",
            "pred_label": labels[row["pred_label"]],
            "true_label": labels[row["true_label"]],
            "is_correct": row["is_correct"],
        }


def summarize_votes(predictions, judgment):
    """ResultStoreで検証済みの生出力を、保存せず表示用に集計する。"""
    labels = ("foraging", "rumination")
    # 全保存時刻を使い、時刻内の同値・得票の同票とも小さいノード番号を選ぶ。
    counts = np.bincount(np.argmax(predictions, axis=1), minlength=2)
    total = len(predictions)
    pred_label = labels[int(counts.argmax())]
    return {
        "source": "saved_predictions",
        "total": total,
        "counts": {label: int(counts[index]) for index, label in enumerate(labels)},
        "ratios": {label: float(counts[index] / total) for index, label in enumerate(labels)},
        "pred_label": pred_label,
        "matches_saved_judgment": (
            pred_label == judgment["pred_label"] if judgment["status"] == "available" else None
        ),
    }
