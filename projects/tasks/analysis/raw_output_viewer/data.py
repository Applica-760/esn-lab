import csv
import json
from pathlib import Path

import numpy as np

from projects.utils.eval.judgment import load_judgment_results
from projects.utils.prediction import load_pred_results


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


class ResultStore:
    """選択中のgroup・parameterのtrain/testだけを保持する。"""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self._selection = None
        self._samples = {}

    def _directory(self, *parts):
        if any(not part or part in (".", "..") or Path(part).name != part for part in parts):
            raise ValueError("group・parameterにはディレクトリ名を指定してください。")
        path = self.root.joinpath(*parts).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("指定したパスは結果ディレクトリの範囲外です。")
        if not path.is_dir():
            raise FileNotFoundError("指定した結果ディレクトリが見つかりません。")
        return path

    def list_conditions(self, group=None, parameter=None):
        if parameter is not None and group is None:
            raise ValueError("parameterを指定する場合はgroupも必要です。")
        directory = self._directory(*([group] if group is not None else []))
        names = sorted(
            p.name
            for p in directory.iterdir()
            if p.is_dir() and p.resolve().is_relative_to(self.root)
        )
        result = {"groups": names} if group is None else {"parameters": names}
        if parameter is not None:
            self._load(group, parameter)
            result = {"folds": sorted({key[0] for key in self._samples})}
        return result

    def _load(self, group, parameter):
        directory = self._directory(group, parameter)
        if self._selection == (group, parameter):
            return
        # 切替後も古い大容量結果を保持しない。
        self._selection = None
        self._samples = {}
        samples = {}
        for mode in ("train", "test"):
            base = directory / f"{mode}_results"
            paths = [base.with_suffix(suffix) for suffix in (".json", ".npz")]
            path = next((p for p in paths if p.is_file()), None)
            if path is None:
                continue
            if not path.resolve().is_relative_to(self.root):
                raise ValueError("結果ファイルは結果ディレクトリ内に置いてください。")
            try:
                if path.suffix == ".json":
                    # 既存JSONローダーのfloat32変換を避け、保存値を保持する。
                    folds = json.loads(path.read_text())
                else:
                    with np.load(path, allow_pickle=False) as archive:
                        lengths = archive["lengths"]
                        if (
                            lengths.ndim != 1
                            or lengths.dtype.kind not in "iu"
                            or (lengths <= 0).any()
                            or archive["ids"].shape != lengths.shape
                            or archive["fold_indices"].shape != lengths.shape
                            or archive["fold_indices"].dtype.kind not in "iu"
                        ):
                            raise ValueError("NPZのサンプル情報が不正です。")
                    folds = load_pred_results(str(base))
                    loaded = [s for f in folds for s in f["results"]]
                    if len(loaded) != len(lengths) or sum(
                        len(s["predictions"]) for s in loaded
                    ) != int(lengths.sum()):
                        raise ValueError("NPZの系列長と保存データが一致しません。")
                if not isinstance(folds, list):
                    raise ValueError("fold一覧が不正です。")
                for fold in folds:
                    index = fold["fold_index"]
                    if type(index) is not int or index < 0:
                        raise ValueError("fold_indexが不正です。")
                    for sample in fold["results"]:
                        sample_id = sample["id"]
                        if not isinstance(sample_id, str) or not sample_id:
                            raise ValueError("サンプルIDが不正です。")
                        key = (index, mode, sample_id)
                        if key in samples:
                            raise ValueError(f"サンプルIDが重複しています: {sample_id}")
                        predictions = np.asarray(sample["predictions"])
                        labels = np.asarray(sample["labels"])
                        if (
                            predictions.ndim != 2
                            or predictions.shape[1] != 2
                            or len(predictions) == 0
                            or labels.shape != predictions.shape
                        ):
                            raise ValueError(f"2ノードの出力・ラベルの形状が不正です: {sample_id}")
                        if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
                            raise ValueError(
                                f"出力またはラベルに欠損・非有限値があります: {sample_id}"
                            )
                        if (
                            not np.isin(labels, [0, 1]).all()
                            or not (labels.sum(axis=1) == 1).all()
                            or not (labels == labels[0]).all()
                        ):
                            raise ValueError(f"単一の真値ラベルを確認できません: {sample_id}")
                        samples[key] = {
                            "predictions": predictions,
                            "true_label": ("foraging", "rumination")[int(labels[0].argmax())],
                        }
            except Exception as exc:
                raise ValueError(f"{mode}の結果を読み込めません: {exc}") from exc
        if not samples:
            raise FileNotFoundError("指定した条件の予測結果が見つかりません。")
        self._samples = samples
        self._selection = (group, parameter)

    def list_samples(self, group, parameter, fold):
        self._load(group, parameter)
        result = [
            {
                "id": sample_id,
                "mode": mode,
                "true_label": sample["true_label"],
                "length": len(sample["predictions"]),
            }
            for (index, mode, sample_id), sample in sorted(self._samples.items())
            if index == fold
        ]
        if not result:
            raise FileNotFoundError("指定したfoldのデータが見つかりません。")
        return result

    def list_sample_folds(self, group, parameter, sample_id):
        self._load(group, parameter)
        result = [
            {
                "fold": fold,
                "modes": [
                    mode for mode in ("train", "test") if (fold, mode, sample_id) in self._samples
                ],
            }
            for fold in sorted({key[0] for key in self._samples})
        ]
        if not any(item["modes"] for item in result):
            raise FileNotFoundError("指定した条件・IDのデータが見つかりません。")
        return result

    def get_sample(self, group, parameter, fold, mode, sample_id):
        self._load(group, parameter)
        try:
            return self._samples[(fold, mode, sample_id)]
        except KeyError:
            raise FileNotFoundError("指定した条件・IDのデータが見つかりません。") from None
