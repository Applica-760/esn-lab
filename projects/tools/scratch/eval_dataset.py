#!/usr/bin/env python3
"""
データセットのnpzファイル構造を確認するスクリプト

Usage:
python projects/tools/scratch/eval_dataset.py dataset/10fold_npy/a/fold_a.npz
"""
import sys
from pathlib import Path
import numpy as np


def inspect_npz(npz_path: Path):
    print(f"=== {npz_path.name} ===\n")
    
    data = np.load(npz_path)
    
    print(f"Keys: {list(data.keys())}\n")
    
    if "num_samples" in data:
        num_samples = int(data["num_samples"])
        print(f"num_samples: {num_samples}\n")
        
        for i in range(min(3, num_samples)):
            print(f"--- Sample {i} ---")
            
            if f"{i}_id" in data:
                sample_id = data[f"{i}_id"]
                print(f"  ID: {sample_id} (type: {type(sample_id).__name__})")
            
            if f"{i}_data" in data:
                sample_data = data[f"{i}_data"]
                print(f"  Data shape: {sample_data.shape}, dtype: {sample_data.dtype}")
                print(f"  Data range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
                if i == 0:
                    print(f"  Data array:\n{sample_data}")
            
            if f"{i}_label" in data:
                sample_label = data[f"{i}_label"]
                print(f"  Label shape: {sample_label.shape}, dtype: {sample_label.dtype}")
                print(f"  Label (class index): {np.argmax(sample_label[:, 0])}")
                print(f"  Label unique values: {np.unique(sample_label)}")
                if i == 0:
                    print(f"  Label array:\n{sample_label}")
            
            print()
        
        if num_samples > 3:
            print(f"... and {num_samples - 3} more samples\n")
    
    data.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_dataset.py <path_to_npz>")
        sys.exit(1)
    
    npz_path = Path(sys.argv[1])
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)
    
    inspect_npz(npz_path)
