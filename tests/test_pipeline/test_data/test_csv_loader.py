"""Tests for CSVDataLoader."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from esn_lab.pipeline.data.csv_loader import CSVDataLoader


@pytest.fixture
def temp_csv_dir():
    """Create a temporary directory with CSV data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        img_dir.mkdir()

        # Create test images for fold_a (2 samples)
        img_a_0 = np.random.randint(0, 256, (5, 10), dtype=np.uint8)
        img_a_1 = np.random.randint(0, 256, (5, 15), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "sample_a_0.png"), img_a_0)
        cv2.imwrite(str(img_dir / "sample_a_1.png"), img_a_1)

        # Create CSV for fold_a
        df_a = pd.DataFrame({
            "file_path": [
                str(img_dir / "sample_a_0.png"),
                str(img_dir / "sample_a_1.png"),
            ],
            "behavior": [0, 1],
        })
        df_a.to_csv(tmpdir / "10fold_a.csv", index=False)

        # Create test images for fold_b (3 samples)
        img_b_0 = np.random.randint(0, 256, (5, 12), dtype=np.uint8)
        img_b_1 = np.random.randint(0, 256, (5, 8), dtype=np.uint8)
        img_b_2 = np.random.randint(0, 256, (5, 20), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "sample_b_0.png"), img_b_0)
        cv2.imwrite(str(img_dir / "sample_b_1.png"), img_b_1)
        cv2.imwrite(str(img_dir / "sample_b_2.png"), img_b_2)

        # Create CSV for fold_b
        df_b = pd.DataFrame({
            "file_path": [
                str(img_dir / "sample_b_0.png"),
                str(img_dir / "sample_b_1.png"),
                str(img_dir / "sample_b_2.png"),
            ],
            "behavior": [2, 0, 1],
        })
        df_b.to_csv(tmpdir / "10fold_b.csv", index=False)

        yield tmpdir


class TestCSVDataLoaderInitialization:
    """Test CSVDataLoader initialization and error handling."""

    def test_init_with_valid_directory(self, temp_csv_dir):
        """Test initialization with a valid directory."""
        loader = CSVDataLoader(temp_csv_dir)
        assert loader.csv_dir == temp_csv_dir

    def test_init_with_nonexistent_directory(self):
        """Test initialization with a non-existent directory."""
        with pytest.raises(FileNotFoundError, match="CSV directory not found"):
            CSVDataLoader("/nonexistent/path")

    def test_init_without_csv_files(self):
        """Test initialization without any CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No 10fold_.*\\.csv files found"):
                CSVDataLoader(tmpdir)


class TestCSVDataLoaderFoldDetection:
    """Test fold detection functionality."""

    def test_detect_available_folds(self, temp_csv_dir):
        """Test detection of available folds."""
        loader = CSVDataLoader(temp_csv_dir)
        folds = loader.get_available_folds()
        assert folds == ["a", "b"]

    def test_get_available_folds_returns_copy(self, temp_csv_dir):
        """Test that get_available_folds returns a copy, not reference."""
        loader = CSVDataLoader(temp_csv_dir)
        folds1 = loader.get_available_folds()
        folds2 = loader.get_available_folds()

        # Modify one list
        folds1.append("z")

        # Check that the other list is unaffected
        assert folds2 == ["a", "b"]
        assert loader.get_available_folds() == ["a", "b"]

    def test_fold_detection_all_letters(self):
        """Test fold detection with all possible fold letters (a-j)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sparse folds
            for ch in "acegi":
                df = pd.DataFrame({
                    "file_path": ["dummy.png"],
                    "behavior": [0],
                })
                df.to_csv(tmpdir / f"10fold_{ch}.csv", index=False)

            loader = CSVDataLoader(tmpdir)
            assert loader.get_available_folds() == ["a", "c", "e", "g", "i"]


class TestCSVDataLoaderFoldSize:
    """Test get_fold_size functionality."""

    def test_get_fold_size(self, temp_csv_dir):
        """Test getting fold size from CSV."""
        loader = CSVDataLoader(temp_csv_dir)
        assert loader.get_fold_size("a") == 2
        assert loader.get_fold_size("b") == 3

    def test_get_fold_size_nonexistent_fold(self, temp_csv_dir):
        """Test getting size of non-existent fold."""
        loader = CSVDataLoader(temp_csv_dir)
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            loader.get_fold_size("z")


class TestCSVDataLoaderImageLoading:
    """Test image loading functionality."""

    def test_load_single_fold(self, temp_csv_dir):
        """Test loading data from a single fold."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data(["a"]))

        assert len(samples) == 2

        # Check first sample
        sample_id, sequence, class_id = samples[0]
        assert sample_id == "sample_a_0"
        assert sequence.shape == (10, 5)  # Transposed from (5, 10)
        assert class_id == 0

        # Check second sample
        sample_id, sequence, class_id = samples[1]
        assert sample_id == "sample_a_1"
        assert sequence.shape == (15, 5)  # Transposed from (5, 15)
        assert class_id == 1

    def test_load_multiple_folds(self, temp_csv_dir):
        """Test loading data from multiple folds."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data(["a", "b"]))

        # Should have 2 + 3 = 5 samples total
        assert len(samples) == 5

        # Check sample IDs
        assert samples[0][0] == "sample_a_0"
        assert samples[1][0] == "sample_a_1"
        assert samples[2][0] == "sample_b_0"
        assert samples[3][0] == "sample_b_1"
        assert samples[4][0] == "sample_b_2"

    def test_image_transpose(self, temp_csv_dir):
        """Test that images are properly transposed to (T, D) format."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data(["a"]))

        # Original image is (5, 10), should be transposed to (10, 5)
        _, sequence, _ = samples[0]
        assert sequence.shape[0] > sequence.shape[1]  # T > D after transpose

    def test_missing_image_file(self, temp_csv_dir):
        """Test error handling when image file is missing."""
        # Create CSV with non-existent image path
        df = pd.DataFrame({
            "file_path": ["/nonexistent/image.png"],
            "behavior": [0],
        })
        df.to_csv(temp_csv_dir / "10fold_c.csv", index=False)

        loader = CSVDataLoader(temp_csv_dir)
        with pytest.raises(FileNotFoundError, match="Failed to read image"):
            list(loader.load_fold_data(["c"]))

    def test_csv_column_validation(self, temp_csv_dir):
        """Test CSV column validation."""
        # Create CSV with wrong columns
        df = pd.DataFrame({
            "wrong_column": ["image.png"],
            "another_wrong": [0],
        })
        df.to_csv(temp_csv_dir / "10fold_c.csv", index=False)

        loader = CSVDataLoader(temp_csv_dir)
        # pandas raises ValueError when usecols don't match actual columns
        with pytest.raises(ValueError, match="Usecols do not match columns"):
            list(loader.load_fold_data(["c"]))

    def test_load_nonexistent_fold(self, temp_csv_dir):
        """Test loading from a non-existent fold."""
        loader = CSVDataLoader(temp_csv_dir)
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            list(loader.load_fold_data(["z"]))


class TestCSVDataLoaderIterator:
    """Test iterator behavior."""

    def test_iterator_is_memory_efficient(self, temp_csv_dir):
        """Test that iterator yields one sample at a time."""
        loader = CSVDataLoader(temp_csv_dir)
        iterator = loader.load_fold_data(["a", "b"])

        # Get first sample
        sample_id, sequence, class_id = next(iterator)
        assert sample_id == "sample_a_0"

        # Get second sample
        sample_id, sequence, class_id = next(iterator)
        assert sample_id == "sample_a_1"

        # Verify we can continue iterating
        remaining_samples = list(iterator)
        assert len(remaining_samples) == 3

    def test_empty_fold_list(self, temp_csv_dir):
        """Test loading with empty fold list."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data([]))
        assert len(samples) == 0

    def test_ragged_sequences(self, temp_csv_dir):
        """Test that sequences can have different lengths."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data(["a", "b"]))

        # Get all sequence lengths
        lengths = [seq.shape[0] for _, seq, _ in samples]

        # Verify different lengths
        assert len(set(lengths)) > 1, "Should have varying sequence lengths"
        assert 10 in lengths
        assert 15 in lengths
        assert 12 in lengths
        assert 8 in lengths
        assert 20 in lengths


class TestCSVDataLoaderMetadata:
    """Test metadata extraction from CSV files."""

    def test_sample_id_extraction(self, temp_csv_dir):
        """Test that sample IDs are correctly extracted from file paths."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data(["a"]))

        # Sample IDs should be the stem (filename without extension)
        sample_ids = [sid for sid, _, _ in samples]
        assert "sample_a_0" in sample_ids
        assert "sample_a_1" in sample_ids

    def test_class_ids_are_integers(self, temp_csv_dir):
        """Test that class IDs are properly converted to integers."""
        loader = CSVDataLoader(temp_csv_dir)
        samples = list(loader.load_fold_data(["b"]))

        class_ids = [cid for _, _, cid in samples]
        assert all(isinstance(cid, int) for cid in class_ids)
        assert class_ids == [2, 0, 1]
