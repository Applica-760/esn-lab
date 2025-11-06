#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_10fold_npy.py

10-fold NPY形式のデータセットの整合性を検証するスクリプト

検証項目:
1. ファイル構造の整合性
2. データ形式の整合性
3. クラスバランスの検証
4. 各foldでのラベルごとのサンプル数の一致
5. 時系列長の検証
6. メタデータとの整合性

Usage:
python tools/dataprep/validate_10fold_npy.py \
    --npy-dir dataset/10fold_npy/ \
    --report dataset/validation_report.txt \
    --verbose
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


class Colors:
    """ターミナル出力用のカラーコード"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ValidationResult:
    """検証結果を格納するクラス"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = True
        self.messages: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.details: Dict = {}
    
    def add_error(self, message: str):
        """エラーを追加"""
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        """警告を追加"""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """情報メッセージを追加"""
        self.messages.append(message)
    
    def add_detail(self, key: str, value):
        """詳細情報を追加"""
        self.details[key] = value


class NPYDatasetValidator:
    """10-fold NPYデータセットの検証クラス"""
    
    EXPECTED_FOLDS = list("abcdefghij")
    EXPECTED_CLASSES = [0, 1, 2]
    
    def __init__(self, npy_dir: Path, verbose: bool = False):
        self.npy_dir = npy_dir
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.metadata: Optional[Dict] = None
        self.fold_data: Dict[str, Dict] = {}
    
    def run_all_validations(self) -> bool:
        """全ての検証を実行"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}10-Fold NPY Dataset Validation{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        print(f"Target directory: {self.npy_dir}\n")
        
        # 検証項目を順次実行
        self.validate_file_structure()
        self.load_metadata()
        self.validate_data_format()
        self.validate_class_balance()
        self.validate_samples_per_class()
        self.validate_time_lengths()
        self.validate_metadata_consistency()
        
        # 結果サマリーを表示
        return self.print_summary()
    
    def validate_file_structure(self):
        """1. ファイル構造の整合性を検証"""
        result = ValidationResult("File Structure")
        
        # NPYディレクトリの存在確認
        if not self.npy_dir.exists():
            result.add_error(f"NPY directory does not exist: {self.npy_dir}")
            self.results.append(result)
            return
        
        # metadata.jsonの存在確認
        metadata_path = self.npy_dir / "metadata.json"
        if not metadata_path.exists():
            result.add_error(f"metadata.json not found in {self.npy_dir}")
        else:
            result.add_info(f"metadata.json found")
        
        # 各foldのNPZファイル存在確認
        missing_folds = []
        found_folds = []
        for fold_id in self.EXPECTED_FOLDS:
            npz_path = self.npy_dir / f"fold_{fold_id}.npz"
            if npz_path.exists():
                found_folds.append(fold_id)
                result.add_info(f"fold_{fold_id}.npz found")
            else:
                missing_folds.append(fold_id)
                result.add_error(f"fold_{fold_id}.npz not found")
        
        result.add_detail("found_folds", found_folds)
        result.add_detail("missing_folds", missing_folds)
        
        self.results.append(result)
        self._print_result(result)
    
    def load_metadata(self):
        """metadata.jsonを読み込み"""
        result = ValidationResult("Metadata Loading")
        
        metadata_path = self.npy_dir / "metadata.json"
        if not metadata_path.exists():
            result.add_error("Cannot load metadata.json - file not found")
            self.results.append(result)
            self._print_result(result)
            return
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            result.add_info(f"Successfully loaded metadata.json")
            result.add_detail("format", self.metadata.get("format", "unknown"))
            result.add_detail("created_at", self.metadata.get("created_at", "unknown"))
        except json.JSONDecodeError as e:
            result.add_error(f"Failed to parse metadata.json: {e}")
        except Exception as e:
            result.add_error(f"Error loading metadata.json: {e}")
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_data_format(self):
        """2. データ形式の整合性を検証"""
        result = ValidationResult("Data Format")
        
        for fold_id in self.EXPECTED_FOLDS:
            npz_path = self.npy_dir / f"fold_{fold_id}.npz"
            if not npz_path.exists():
                continue
            
            try:
                data = np.load(npz_path, allow_pickle=False)
                
                # num_samplesの確認
                if "num_samples" not in data:
                    result.add_error(f"fold_{fold_id}: 'num_samples' key not found")
                    continue
                
                num_samples = int(data["num_samples"])
                result.add_info(f"fold_{fold_id}: {num_samples} samples")
                
                # 各サンプルのキー構造を確認
                sample_data = []
                sample_ids = []
                class_ids = []
                feature_dims = []
                
                for i in range(num_samples):
                    data_key = f"sample_{i}_data"
                    id_key = f"sample_{i}_id"
                    class_key = f"sample_{i}_class"
                    
                    # キーの存在確認
                    if data_key not in data:
                        result.add_error(f"fold_{fold_id}: '{data_key}' not found")
                        continue
                    if id_key not in data:
                        result.add_error(f"fold_{fold_id}: '{id_key}' not found")
                    if class_key not in data:
                        result.add_error(f"fold_{fold_id}: '{class_key}' not found")
                    
                    # データ形状の確認
                    arr = data[data_key]
                    if arr.ndim != 2:
                        result.add_error(f"fold_{fold_id}, sample_{i}: Expected 2D array, got {arr.ndim}D")
                        continue
                    
                    sample_data.append(arr)
                    sample_ids.append(str(data[id_key]))
                    class_ids.append(int(data[class_key]))
                    feature_dims.append(arr.shape[1])
                
                # feature_dimの統一性確認
                if len(set(feature_dims)) > 1:
                    result.add_error(f"fold_{fold_id}: Inconsistent feature dimensions: {set(feature_dims)}")
                else:
                    result.add_info(f"fold_{fold_id}: Uniform feature_dim = {feature_dims[0]}")
                
                # fold_dataに格納
                self.fold_data[fold_id] = {
                    "num_samples": num_samples,
                    "sample_data": sample_data,
                    "sample_ids": sample_ids,
                    "class_ids": class_ids,
                    "feature_dim": feature_dims[0] if feature_dims else None,
                }
                
            except Exception as e:
                result.add_error(f"fold_{fold_id}: Error loading NPZ - {e}")
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_class_balance(self):
        """3. クラスバランスの検証"""
        result = ValidationResult("Class Balance")
        
        if not self.fold_data:
            result.add_error("No fold data loaded")
            self.results.append(result)
            self._print_result(result)
            return
        
        fold_class_counts = {}
        
        for fold_id, data in self.fold_data.items():
            class_ids = data["class_ids"]
            counter = Counter(class_ids)
            fold_class_counts[fold_id] = counter
            
            # 期待されるクラスの存在確認
            for expected_class in self.EXPECTED_CLASSES:
                if expected_class not in counter:
                    result.add_error(f"fold_{fold_id}: Class {expected_class} not found")
            
            # 各foldのクラス分布を表示
            if self.verbose:
                result.add_info(f"fold_{fold_id}: {dict(counter)}")
            
            # クラス間のバランス確認（同一fold内）
            counts = [counter.get(c, 0) for c in self.EXPECTED_CLASSES]
            if len(set(counts)) > 1:
                result.add_warning(f"fold_{fold_id}: Unbalanced classes - {dict(counter)}")
            else:
                result.add_info(f"fold_{fold_id}: Balanced classes ({counts[0]} each)")
        
        result.add_detail("fold_class_counts", fold_class_counts)
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_samples_per_class(self):
        """4. 各foldでラベルごとのサンプル数が一致するかを検証"""
        result = ValidationResult("Samples Per Class Consistency")
        
        if not self.fold_data:
            result.add_error("No fold data loaded")
            self.results.append(result)
            self._print_result(result)
            return
        
        # 各クラスごとに全fold間でサンプル数を収集
        class_sample_counts = defaultdict(list)
        
        for fold_id, data in self.fold_data.items():
            counter = Counter(data["class_ids"])
            for class_id in self.EXPECTED_CLASSES:
                count = counter.get(class_id, 0)
                class_sample_counts[class_id].append((fold_id, count))
        
        # 各クラスについて、全foldで一致しているか確認
        all_consistent = True
        for class_id in self.EXPECTED_CLASSES:
            counts = class_sample_counts[class_id]
            count_values = [c for _, c in counts]
            
            if len(set(count_values)) == 1:
                result.add_info(f"Class {class_id}: Consistent across all folds ({count_values[0]} samples each)")
            else:
                all_consistent = False
                result.add_error(f"Class {class_id}: Inconsistent sample counts across folds")
                for fold_id, count in counts:
                    result.add_error(f"  fold_{fold_id}: {count} samples")
        
        if all_consistent:
            # 全クラスで一致している場合、サンプル数の表を表示
            result.add_info("\nSummary:")
            for class_id in self.EXPECTED_CLASSES:
                count = class_sample_counts[class_id][0][1]
                result.add_info(f"  Class {class_id}: {count} samples per fold")
        
        result.add_detail("class_sample_counts", dict(class_sample_counts))
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_time_lengths(self):
        """5. 時系列長の検証"""
        result = ValidationResult("Time Length Statistics")
        
        if not self.fold_data:
            result.add_error("No fold data loaded")
            self.results.append(result)
            self._print_result(result)
            return
        
        for fold_id, data in self.fold_data.items():
            sample_data = data["sample_data"]
            time_lengths = [arr.shape[0] for arr in sample_data]
            
            min_t = min(time_lengths)
            max_t = max(time_lengths)
            mean_t = np.mean(time_lengths)
            
            result.add_info(f"fold_{fold_id}: T_min={min_t}, T_max={max_t}, T_mean={mean_t:.1f}")
            
            # metadataとの比較（存在する場合）
            if self.metadata and "conversion_stats" in self.metadata:
                for stat in self.metadata["conversion_stats"]:
                    if stat.get("fold_id") == fold_id:
                        meta_min = stat.get("min_time_length")
                        meta_max = stat.get("max_time_length")
                        meta_mean = stat.get("mean_time_length")
                        
                        if meta_min != min_t:
                            result.add_warning(f"fold_{fold_id}: min_time_length mismatch (actual={min_t}, metadata={meta_min})")
                        if meta_max != max_t:
                            result.add_warning(f"fold_{fold_id}: max_time_length mismatch (actual={max_t}, metadata={meta_max})")
                        if abs(meta_mean - mean_t) > 0.1:
                            result.add_warning(f"fold_{fold_id}: mean_time_length mismatch (actual={mean_t:.1f}, metadata={meta_mean:.1f})")
        
        self.results.append(result)
        self._print_result(result)
    
    def validate_metadata_consistency(self):
        """6. メタデータとの整合性を検証"""
        result = ValidationResult("Metadata Consistency")
        
        if not self.metadata:
            result.add_warning("metadata.json not loaded, skipping consistency check")
            self.results.append(result)
            self._print_result(result)
            return
        
        # foldsセクションとの比較
        if "folds" in self.metadata:
            metadata_folds = self.metadata["folds"]
            
            for fold_id, data in self.fold_data.items():
                fold_key = f"fold_{fold_id}"
                
                if fold_key not in metadata_folds:
                    result.add_error(f"fold_{fold_id}: Not found in metadata.folds")
                    continue
                
                metadata_mapping = metadata_folds[fold_key]
                
                # sample_idとclass_idの対応を確認
                for sample_id, class_id in zip(data["sample_ids"], data["class_ids"]):
                    if sample_id in metadata_mapping:
                        meta_class = metadata_mapping[sample_id]
                        if meta_class != class_id:
                            result.add_error(f"fold_{fold_id}, {sample_id}: class_id mismatch (actual={class_id}, metadata={meta_class})")
                    else:
                        result.add_warning(f"fold_{fold_id}, {sample_id}: Not found in metadata")
                
                result.add_info(f"fold_{fold_id}: Metadata mapping validated")
        
        # conversion_statsセクションの検証
        if "conversion_stats" in self.metadata:
            for stat in self.metadata["conversion_stats"]:
                fold_id = stat.get("fold_id")
                if fold_id in self.fold_data:
                    actual_num = self.fold_data[fold_id]["num_samples"]
                    meta_num = stat.get("num_samples")
                    
                    if actual_num != meta_num:
                        result.add_error(f"fold_{fold_id}: num_samples mismatch (actual={actual_num}, metadata={meta_num})")
        
        self.results.append(result)
        self._print_result(result)
    
    def _print_result(self, result: ValidationResult):
        """個別の検証結果を表示"""
        status = f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}" if result.passed else f"{Colors.FAIL}✗ FAILED{Colors.ENDC}"
        print(f"\n{Colors.BOLD}[{result.test_name}]{Colors.ENDC} {status}")
        
        if self.verbose or not result.passed:
            for msg in result.messages:
                print(f"  {Colors.OKCYAN}ℹ{Colors.ENDC} {msg}")
            for warn in result.warnings:
                print(f"  {Colors.WARNING}⚠{Colors.ENDC} {warn}")
            for err in result.errors:
                print(f"  {Colors.FAIL}✗{Colors.ENDC} {err}")
    
    def print_summary(self) -> bool:
        """検証結果のサマリーを表示"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Validation Summary{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_errors = sum(len(r.errors) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)
        
        print(f"Total tests: {len(self.results)}")
        print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Errors: {total_errors}{Colors.ENDC}")
        print(f"{Colors.WARNING}Warnings: {total_warnings}{Colors.ENDC}\n")
        
        if failed == 0 and total_errors == 0:
            print(f"{Colors.OKGREEN}{Colors.BOLD}✓ All validations passed!{Colors.ENDC}\n")
            return True
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}✗ Validation failed - please review the errors above.{Colors.ENDC}\n")
            return False
    
    def generate_report(self, report_path: Path):
        """検証結果をテキストファイルに出力"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("10-Fold NPY Dataset Validation Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write(f"Target directory: {self.npy_dir}\n\n")
            
            for result in self.results:
                f.write(f"\n{'='*70}\n")
                f.write(f"[{result.test_name}] {'PASSED' if result.passed else 'FAILED'}\n")
                f.write(f"{'='*70}\n\n")
                
                if result.messages:
                    f.write("Info:\n")
                    for msg in result.messages:
                        f.write(f"  ℹ {msg}\n")
                    f.write("\n")
                
                if result.warnings:
                    f.write("Warnings:\n")
                    for warn in result.warnings:
                        f.write(f"  ⚠ {warn}\n")
                    f.write("\n")
                
                if result.errors:
                    f.write("Errors:\n")
                    for err in result.errors:
                        f.write(f"  ✗ {err}\n")
                    f.write("\n")
                
                if result.details:
                    f.write("Details:\n")
                    for key, value in result.details.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            # サマリー
            f.write("\n" + "="*70 + "\n")
            f.write("Summary\n")
            f.write("="*70 + "\n\n")
            
            passed = sum(1 for r in self.results if r.passed)
            failed = len(self.results) - passed
            total_errors = sum(len(r.errors) for r in self.results)
            total_warnings = sum(len(r.warnings) for r in self.results)
            
            f.write(f"Total tests: {len(self.results)}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Errors: {total_errors}\n")
            f.write(f"Warnings: {total_warnings}\n")
        
        print(f"\n{Colors.OKBLUE}Report saved to: {report_path}{Colors.ENDC}")


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Validate 10-fold NPY dataset integrity"
    )
    parser.add_argument(
        "--npy-dir",
        type=str,
        required=True,
        help="Directory containing fold_*.npz files and metadata.json",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save validation report (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information for all tests",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    npy_dir = Path(args.npy_dir).expanduser().resolve()
    
    if not npy_dir.exists():
        print(f"{Colors.FAIL}Error: Directory not found: {npy_dir}{Colors.ENDC}")
        sys.exit(1)
    
    # 検証実行
    validator = NPYDatasetValidator(npy_dir, verbose=args.verbose)
    success = validator.run_all_validations()
    
    # レポート出力
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        validator.generate_report(report_path)
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
