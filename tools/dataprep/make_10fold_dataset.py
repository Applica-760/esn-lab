#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_10fold_datasets_with_behavior.py

指定のCSV（列: file_path, converted_300, uniform_flag, image_width_px）を読み込み、
「converted_300 が 0/1/2 のいずれかで全一致」かつ「image_width_px >= 8000」のデータのみを対象に、
ラベル（0/1/2）ごとの最小件数に合わせて層別サンプリングしたバランスデータを、
乱数シードを固定した10通り（a〜j）で作成して出力します。

出力CSVには元の列に加えて、最右列に 'behavior' 列（各行の一様ラベル 0/1/2）を追加します。


python tools/dataprep/make_10fold_dataset.py \
  --csv data/get_300seqs.csv \
  --out dataset/10fold.csv \
  --min_width 8000 \
  --seeds_base 20251005


"""
from __future__ import annotations
import argparse
import string
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _extract_uniform_label(seq: str) -> int:
    """converted_300 が '0','1','2' の同一文字のみで構成されているかを検証し、その数字を返す。"""
    if not isinstance(seq, str) or len(seq) == 0:
        raise ValueError("converted_300 が空または非文字列です。")
    uniq = set(seq)
    if len(uniq) != 1:
        raise ValueError("uniform_flag==1 のはずなのに converted_300 が一様ではありません。")
    ch = next(iter(uniq))
    if ch not in {"0", "1", "2"}:
        raise ValueError("converted_300 が '0','1','2' 以外の文字を含みます。")
    return int(ch)


def build_balanced_once(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """ラベルごとの最小件数に合わせて、各ラベルから同数をランダム抽出して結合し、シャッフルして返す。"""
    # ラベル列を一時付与
    labels = []
    for s in df["converted_300"].tolist():
        labels.append(_extract_uniform_label(s))
    tmp = df.copy()
    tmp["__label"] = labels

    # 各ラベル件数
    counts = tmp["__label"].value_counts().reindex([0, 1, 2], fill_value=0)
    min_num = int(counts.min())

    # ラベルごとに min_num サンプルを抽出
    rng = np.random.default_rng(seed=random_state)
    parts: List[pd.DataFrame] = []
    for lab in [0, 1, 2]:
        sub = tmp[tmp["__label"] == lab]
        if len(sub) < min_num:
            raise RuntimeError(f"ラベル {lab} の件数が min_num({min_num}) より少ないため抽出できません。")
        idx = rng.choice(sub.index.to_numpy(), size=min_num, replace=False)
        parts.append(sub.loc[idx])

    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 最右に behavior を付与（__label を改名して末尾へ）
    behavior = balanced["__label"].astype(int)
    balanced = balanced.drop(columns=["__label"]).copy()
    balanced["behavior"] = behavior.to_numpy()

    return balanced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="入力CSVパス（列: file_path, converted_300, uniform_flag, image_width_px）")
    ap.add_argument("--out", required=True, help="出力CSVのベース名（例: data/10fold.csv -> data/10fold_*.csv として a〜j を生成）")
    ap.add_argument("--min_width", type=int, default=8000, help="画像の横幅 px の下限（既定: 8000）")
    ap.add_argument("--seeds_base", type=int, default=20251005, help="再現性用のベース乱数シード（既定: 20251005）")
    args = ap.parse_args()

    in_path = Path(args.csv)
    base_out = Path(args.out)

    # 入力
    df = pd.read_csv(in_path, dtype={"converted_300": str})

    # 条件フィルタ：uniform_flag==1 かつ image_width_px >= min_width
    cond = (df["uniform_flag"] == 1) & (df["image_width_px"] >= args.min_width)
    df_ok = df.loc[cond].reset_index(drop=True)
    if df_ok.empty:
        print("条件に合致する行がありませんでした。", file=sys.stderr)
        sys.exit(1)

    # a〜j の10パターンで作成
    suffixes = list(string.ascii_lowercase[:10])  # a b c ... j
    for i, suf in enumerate(suffixes):
        seed_i = args.seeds_base + i  # 連番で固定
        balanced = build_balanced_once(df_ok, random_state=seed_i)

        # 出力ファイル名の決定（ベースの拡張子直前に _{a..j} を入れる）
        out_path = base_out
        if base_out.suffix:
            out_path = base_out.with_name(base_out.stem + f"_{suf}" + base_out.suffix)
        else:
            out_path = base_out.with_name(base_out.name + f"_{suf}.csv")

        balanced.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{suf}] 行数 {len(balanced)} を {out_path} に保存（seed={seed_i}）。")

    print("完了。")

if __name__ == "__main__":
    main()