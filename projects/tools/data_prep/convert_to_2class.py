#!/usr/bin/env python3
"""
3クラスNPZデータセットから other(index=0) を除外し、2クラス（foraging=0, rumination=1）に変換する。

Usage:
uv run projects/tools/data_prep/convert_to_2class.py \
  --input-npz dataset/10fold_npy \
  --output-npz dataset/10fold_npy_2class
"""

import argparse
from pathlib import Path

import numpy as np


def is_other(label: np.ndarray) -> bool:
    """ラベルが other(index=0) かどうかを判定する。"""
    return int(np.argmax(label[0])) == 0


def remap_label(label: np.ndarray) -> np.ndarray:
    """(timesteps, 3) → (timesteps, 2): index1=foraging, index2=rumination を 0,1 に再採番。"""
    return label[:, 1:]


def convert_npz(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """
    1つのNPZファイルを変換して保存する。
    Returns: (元サンプル数, 保持サンプル数)
    """
    with np.load(src_path, allow_pickle=True) as src:
        num_samples = int(src["num_samples"])
        out = {}
        kept = 0
        for i in range(num_samples):
            label = src[f"{i}_label"]
            if is_other(label):
                continue
            out[f"{kept}_id"] = src[f"{i}_id"]
            out[f"{kept}_data"] = src[f"{i}_data"]
            out[f"{kept}_label"] = remap_label(label)
            kept += 1
        out["num_samples"] = kept

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **out)
    return num_samples, kept


def convert_dataset(src_dir: Path, dst_dir: Path) -> None:
    """
    src_dir 配下の全グループ（a〜j）を変換して dst_dir に保存する。
    src と dst が同一パスの場合はオリジナル保護のため即時エラー。
    """
    if src_dir.resolve() == dst_dir.resolve():
        raise ValueError(f"input-npz と output-npz が同一パスです: {src_dir}")

    group_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir())
    if not group_dirs:
        raise FileNotFoundError(f"サブディレクトリが見つかりません: {src_dir}")

    total_src = 0
    total_kept = 0

    for group_dir in group_dirs:
        npz_files = sorted(group_dir.glob("fold_*.npz"))
        for npz_file in npz_files:
            dst_path = dst_dir / group_dir.name / npz_file.name
            n_src, n_kept = convert_npz(npz_file, dst_path)
            total_src += n_src
            total_kept += n_kept
            print(f"  {group_dir.name}/{npz_file.name}: {n_src} → {n_kept} samples")

    print(f"\n変換完了: 合計 {total_src} → {total_kept} samples")
    print(f"  除外（other）: {total_src - total_kept} samples")


def verify_output(dst_dir: Path) -> None:
    """変換後のデータ構造とラベルを抜き取り確認する。"""
    print("\n--- 検証: 変換後NPZのサンプル確認 ---")

    sample_npz = next(dst_dir.glob("*/fold_a.npz"), None)
    if sample_npz is None:
        print("検証用ファイルが見つかりませんでした")
        return

    print(f"対象ファイル: {sample_npz}")
    with np.load(sample_npz, allow_pickle=True) as f:
        num_samples = int(f["num_samples"])
        print(f"  num_samples: {num_samples}")

        for i in range(min(3, num_samples)):
            data = f[f"{i}_data"]
            label = f[f"{i}_label"]
            label_class = int(np.argmax(label[0]))
            print(f"  sample[{i}]  data.shape={data.shape}  label.shape={label.shape}"
                  f"  label[0]={label[0]}  → class={label_class}")

    print("\n--- 検証: ラベル次元が全サンプルで (timesteps, 2) であることを確認 ---")
    errors = 0
    for npz_path in dst_dir.glob("*/fold_*.npz"):
        with np.load(npz_path, allow_pickle=True) as f:
            n = int(f["num_samples"])
            for i in range(n):
                label = f[f"{i}_label"]
                if label.shape[1] != 2:
                    print(f"  NG: {npz_path} sample[{i}] label.shape={label.shape}")
                    errors += 1
                if int(np.argmax(label[0])) not in {0, 1}:
                    print(f"  NG: {npz_path} sample[{i}] unexpected class {np.argmax(label[0])}")
                    errors += 1

    if errors == 0:
        print("  全サンプル OK (shape=(timesteps, 2), class ∈ {0, 1})")
    else:
        print(f"  {errors} 件の異常を検出")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True, help="3クラスNPZデータセットのルートディレクトリ")
    parser.add_argument("--output-npz", required=True, help="2クラスNPZ出力先ディレクトリ（既存と別パスを指定）")
    args = parser.parse_args()

    src_dir = Path(args.input_npz)
    dst_dir = Path(args.output_npz)

    convert_dataset(src_dir, dst_dir)
    verify_output(dst_dir)


if __name__ == "__main__":
    main()
