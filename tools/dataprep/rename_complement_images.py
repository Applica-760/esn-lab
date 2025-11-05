#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path
from shutil import copy2, move

"""
目的:
  親ディレクトリ配下の 20210508, 20210509, 20210510, 20210511 を走査し、
  例: 20210508/064648S1_ffmpeg.jpg → 出力先/08_064648_ffmpeg.jpg に
  リネームしたファイルを出力する（既定はコピー、--move で移動）。

仕様:
  - ディレクトリ末尾2桁を日付(dd)として使用 (例: "20210508" → "08")
  - ファイル名の先頭6桁を時刻(HHMMSS)として使用 (例: "064648S1_ffmpeg.jpg" → "064648")
  - 時刻直後〜最初の '_' までに挟まる不要文字列 (例: 'S1') を削除
    正規化後のベース名: "{dd}_{HHMMSS}_{rest}"
  - 破壊的操作なし（既定は copy）。--move 指定で移動。
  - 出力先に同名が存在する場合はスキップ（安全側）。--overwrite で上書き可能。

想定入力例:
  064648S1_ffmpeg.jpg, 120001X_ffmpeg.jpg, 235959_ffmpeg.jpg 等


python tools/dataprep/rename_images.py \
  --alt-parent-dir data/complement \
  --out-dir data/complements

  
"""

# 先頭6桁の時刻, 続いて '_' までの不要文字列, その後ろの残り
ALT_NAME_RE = re.compile(r'^(?P<time6>\d{6})[^_]*_(?P<rest>.+)$', re.IGNORECASE)

# ディレクトリ名末尾2桁
DIR_DD_RE = re.compile(r'(\d{2})$')

def extract_dd_from_dirname(dirname: str) -> str | None:
    m = DIR_DD_RE.search(dirname)
    return m.group(1) if m else None

def normalize_name(src_name: str, dd: str) -> str | None:
    """
    '064648S1_ffmpeg.jpg' -> '08_064648_ffmpeg.jpg' （dd='08' の場合）
    条件を満たさない名前は None
    """
    m = ALT_NAME_RE.match(src_name)
    if not m:
        return None
    t6 = m.group('time6')
    rest = m.group('rest')
    return f"{dd}_{t6}_{rest}"

def find_target_dirs(parent: Path, days: list[str] | None) -> list[Path]:
    dirs = []
    if days:
        suffixes = set(days)
        for d in parent.iterdir():
            if d.is_dir():
                dd = extract_dd_from_dirname(d.name)
                if dd and dd in suffixes:
                    dirs.append(d)
    else:
        for d in parent.iterdir():
            if d.is_dir() and extract_dd_from_dirname(d.name):
                dirs.append(d)
    return sorted(dirs)

def main():
    ap = argparse.ArgumentParser(description="Rename and export images from 202105xx dirs to 'DD_HHMMSS_rest.ext'.")
    ap.add_argument("--alt-parent-dir", type=Path, required=True,
                    help="20210508〜20210511 が並ぶ親ディレクトリ")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="出力先ディレクトリ（なければ作成）")
    ap.add_argument("--days", type=str, default="",
                    help='対象日付を2桁カンマ区切りで指定 (例: "08,09,10,11")。未指定なら自動抽出。')
    ap.add_argument("--exts", type=str, default=".jpg",
                    help='対象拡張子 (カンマ区切り, 例: ".jpg,.jpeg")。大文字小文字は無視。')
    ap.add_argument("--move", action="store_true", help="コピーではなく移動する")
    ap.add_argument("--overwrite", action="store_true", help="出力先に存在しても上書きする")

    args = ap.parse_args()
    parent: Path = args.alt_parent_dir
    out_dir: Path = args.out_dir
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    days = [d.strip() for d in args.days.split(",") if d.strip()] if args.days.strip() else None

    if not parent.exists() or not parent.is_dir():
        print(f"[ERROR] 親ディレクトリが見つかりません: {parent}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    target_dirs = find_target_dirs(parent, days)
    if not target_dirs:
        print(f"[WARN] 対象ディレクトリが見つかりません（days={days}）: {parent}")
        sys.exit(0)

    total_in = 0
    total_out = 0
    skipped_name = 0
    skipped_exists = 0

    for d in target_dirs:
        dd = extract_dd_from_dirname(d.name)
        if not dd:
            continue

        # 拡張子フィルタで走査
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue

            total_in += 1
            new_name = normalize_name(p.name, dd)
            if not new_name:
                skipped_name += 1
                print(f"[WARN] 名称規則に合わないためスキップ: {p.name}")
                continue

            dst = out_dir / new_name
            if dst.exists() and not args.overwrite:
                skipped_exists += 1
                continue

            try:
                if args.move:
                    move(str(p), str(dst))
                else:
                    copy2(p, dst)
                total_out += 1
            except Exception as e:
                print(f"[WARN] 出力に失敗: {p} -> {dst} ({e})")

    print(f"[INFO] 入力検査: {total_in} 件")
    print(f"[INFO] 出力成功: {total_out} 件")
    print(f"[INFO] 形式不一致スキップ: {skipped_name} 件, 既存衝突スキップ: {skipped_exists} 件 (上書きは --overwrite)")

if __name__ == "__main__":
    main()