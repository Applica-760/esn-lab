#!/usr/bin/env python3
"""
python tools/dataprep/convert_img2binary.py \
	-i /home/takumi/share/esn-lab/data/complements \
	-o /home/takumi/share/esn-lab/dataset/complements-binary

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2  # type: ignore
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def read_grayscale(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError(f"画像の読み込みに失敗しました: {path}")
	# OpenCVのIMREAD_GRAYSCALEは2次元のndarrayを返す
	if img.ndim != 2:
		# 理論上起こりにくいが、堅牢性のため再変換
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img


def convert_images(input_dir: Path, output_dir: Path) -> Tuple[int, int]:
	output_dir.mkdir(parents=True, exist_ok=True)

	processed = 0
	skipped = 0

	# 非再帰で列挙
	for entry in sorted(input_dir.iterdir()):
		if not entry.is_file():
			continue

		if not is_image_file(entry):
			skipped += 1
			continue

		try:
			img = read_grayscale(entry)
			out_path = output_dir / f"{entry.stem}.npy"
			np.save(out_path, img)
			processed += 1
		except Exception as e: 
			print(f"[SKIP] {entry.name}: {e}")
			skipped += 1

	return processed, skipped


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"グレースケール画像を .npy に一括変換します（非再帰）。"
		)
	)
	parser.add_argument(
		"-i",
		"--input",
		required=True,
		type=Path,
		help="入力ディレクトリ（グレースケール画像群）",
	)
	parser.add_argument(
		"-o",
		"--output",
		required=True,
		type=Path,
		help="出力ディレクトリ（.npyを保存）",
	)
	return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
	args = parse_args(argv)
	input_dir: Path = args.input
	output_dir: Path = args.output

	if not input_dir.exists() or not input_dir.is_dir():
		print(f"入力ディレクトリが見つかりません: {input_dir}")
		return 1

	processed, skipped = convert_images(input_dir, output_dir)
	print(
		f"完了: {processed} 件変換, {skipped} 件スキップ -> 出力: {output_dir}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

