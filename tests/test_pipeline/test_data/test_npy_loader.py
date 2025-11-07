"""Tests for NPYDataLoader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from esn_lab.pipeline.data.npy_loader import NPYDataLoader


@pytest.fixture
def temp_npy_dir():
    """Create a temporary directory with NPY data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create metadata.json
        metadata = {
            "num_classes": 3,
            "folds": {
                "fold_a": ["sample_0", "sample_1"],
                "fold_b": ["sample_2", "sample_3", "sample_4"],
            }
        }
        with open(tmpdir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Create fold_a.npz with 2 samples
        fold_a_data = {
            "num_samples": 2,
            "fold_id": "a",
            "sample_0_id": "sample_0",
            "sample_0_data": np.random.randn(10, 5).astype(np.float32),
            "sample_0_class": 0,
            "sample_1_id": "sample_1",
            "sample_1_data": np.random.randn(15, 5).astype(np.float32),
            "sample_1_class": 1,
        }
        np.savez(tmpdir / "fold_a.npz", **fold_a_data)

        # Create fold_b.npz with 3 samples
        fold_b_data = {
            "num_samples": 3,
            "fold_id": "b",
            "sample_0_id": "sample_2",
            "sample_0_data": np.random.randn(12, 5).astype(np.float32),
            "sample_0_class": 2,
            "sample_1_id": "sample_3",
            "sample_1_data": np.random.randn(8, 5).astype(np.float32),
            "sample_1_class": 0,
            "sample_2_id": "sample_4",
            "sample_2_data": np.random.randn(20, 5).astype(np.float32),
            "sample_2_class": 1,
        }
        np.savez(tmpdir / "fold_b.npz", **fold_b_data)

        yield tmpdir


class TestNPYDataLoaderInitialization:
    """Test NPYDataLoader initialization and error handling."""

    def test_init_with_valid_directory(self, temp_npy_dir):
        """Test initialization with a valid directory."""
        loader = NPYDataLoader(temp_npy_dir)
        assert loader.npy_dir == temp_npy_dir
        assert loader.metadata is not None
        assert loader.metadata["num_classes"] == 3

    def test_init_with_nonexistent_directory(self):
        """Test initialization with a non-existent directory."""
        with pytest.raises(FileNotFoundError, match="NPY directory not found"):
            NPYDataLoader("/nonexistent/path")

    def test_init_without_metadata(self):
        """Test initialization without metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create fold file without metadata
            np.savez(tmpdir / "fold_a.npz", num_samples=1)

            with pytest.raises(FileNotFoundError, match="metadata.json not found"):
                NPYDataLoader(tmpdir)

    def test_init_without_folds(self):
        """Test initialization without any fold files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Create only metadata
            metadata = {"num_classes": 3}
            with open(tmpdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            with pytest.raises(ValueError, match="No fold_.*\\.npz files found"):
                NPYDataLoader(tmpdir)


class TestNPYDataLoaderFoldDetection:
    """Test fold detection functionality."""

    def test_detect_available_folds(self, temp_npy_dir):
        """Test detection of available folds."""
        loader = NPYDataLoader(temp_npy_dir)
        folds = loader.get_available_folds()
        assert folds == ["a", "b"]

    def test_get_available_folds_returns_copy(self, temp_npy_dir):
        """Test that get_available_folds returns a copy, not reference."""
        loader = NPYDataLoader(temp_npy_dir)
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

            # Create metadata
            metadata = {"num_classes": 2}
            with open(tmpdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            # Create folds a through j
            expected_folds = []
            for ch in "acegi":  # Create sparse folds
                fold_data = {
                    "num_samples": 1,
                    "sample_0_id": "sample",
                    "sample_0_data": np.zeros((5, 3)),
                    "sample_0_class": 0,
                }
                np.savez(tmpdir / f"fold_{ch}.npz", **fold_data)
                expected_folds.append(ch)

            loader = NPYDataLoader(tmpdir)
            assert loader.get_available_folds() == sorted(expected_folds)


class TestNPYDataLoaderFoldSize:
    """Test get_fold_size functionality."""

    def test_get_fold_size_from_metadata(self, temp_npy_dir):
        """Test getting fold size from metadata."""
        loader = NPYDataLoader(temp_npy_dir)
        assert loader.get_fold_size("a") == 2
        assert loader.get_fold_size("b") == 3

    def test_get_fold_size_from_npz_file(self):
        """Test getting fold size from NPZ file when metadata is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create metadata without fold info
            metadata = {"num_classes": 2}
            with open(tmpdir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            # Create fold file
            fold_data = {
                "num_samples": 5,
                "sample_0_id": "s0",
                "sample_0_data": np.zeros((5, 3)),
                "sample_0_class": 0,
            }
            np.savez(tmpdir / "fold_a.npz", **fold_data)

            loader = NPYDataLoader(tmpdir)
            assert loader.get_fold_size("a") == 5

    def test_get_fold_size_nonexistent_fold(self, temp_npy_dir):
        """Test getting size of non-existent fold."""
        loader = NPYDataLoader(temp_npy_dir)
        with pytest.raises(FileNotFoundError, match="NPZ file not found"):
            loader.get_fold_size("z")


class TestNPYDataLoaderIterator:
    """Test iterator behavior and data loading."""

    def test_load_single_fold(self, temp_npy_dir):
        """Test loading data from a single fold."""
        loader = NPYDataLoader(temp_npy_dir)
        samples = list(loader.load_fold_data(["a"]))

        assert len(samples) == 2

        # Check first sample
        sample_id, sequence, class_id = samples[0]
        assert sample_id == "sample_0"
        assert sequence.shape == (10, 5)
        assert class_id == 0

        # Check second sample
        sample_id, sequence, class_id = samples[1]
        assert sample_id == "sample_1"
        assert sequence.shape == (15, 5)
        assert class_id == 1

    def test_load_multiple_folds(self, temp_npy_dir):
        """Test loading data from multiple folds."""
        loader = NPYDataLoader(temp_npy_dir)
        samples = list(loader.load_fold_data(["a", "b"]))

        # Should have 2 + 3 = 5 samples total
        assert len(samples) == 5

        # Check samples from fold_a
        assert samples[0][0] == "sample_0"
        assert samples[1][0] == "sample_1"

        # Check samples from fold_b
        assert samples[2][0] == "sample_2"
        assert samples[3][0] == "sample_3"
        assert samples[4][0] == "sample_4"

    def test_iterator_returns_copies(self, temp_npy_dir):
        """Test that iterator returns proper copies, not mmap references."""
        loader = NPYDataLoader(temp_npy_dir)

        # Load data
        samples = []
        for sample_id, sequence, class_id in loader.load_fold_data(["a"]):
            samples.append((sample_id, sequence.copy(), class_id))

        # NPZ file is now closed, but data should still be accessible
        assert len(samples) == 2
        assert samples[0][1].shape == (10, 5)

        # Verify we can still access and modify the data
        samples[0][1][0, 0] = 999.0
        assert samples[0][1][0, 0] == 999.0

    def test_load_nonexistent_fold(self, temp_npy_dir):
        """Test loading from a non-existent fold."""
        loader = NPYDataLoader(temp_npy_dir)

        with pytest.raises(FileNotFoundError, match="NPZ file not found"):
            list(loader.load_fold_data(["z"]))

    def test_iterator_is_memory_efficient(self, temp_npy_dir):
        """Test that iterator yields one sample at a time (not loading all)."""
        loader = NPYDataLoader(temp_npy_dir)
        iterator = loader.load_fold_data(["a", "b"])

        # Get first sample
        sample_id, sequence, class_id = next(iterator)
        assert sample_id == "sample_0"

        # Get second sample
        sample_id, sequence, class_id = next(iterator)
        assert sample_id == "sample_1"

        # Verify we can continue iterating
        remaining_samples = list(iterator)
        assert len(remaining_samples) == 3

    def test_empty_fold_list(self, temp_npy_dir):
        """Test loading with empty fold list."""
        loader = NPYDataLoader(temp_npy_dir)
        samples = list(loader.load_fold_data([]))
        assert len(samples) == 0

    def test_ragged_sequences(self, temp_npy_dir):
        """Test that sequences can have different lengths (ragged array support)."""
        loader = NPYDataLoader(temp_npy_dir)
        samples = list(loader.load_fold_data(["a", "b"]))

        # Get all sequence lengths
        lengths = [seq.shape[0] for _, seq, _ in samples]

        # Verify different lengths
        assert len(set(lengths)) > 1, "Should have varying sequence lengths"
        assert 10 in lengths  # From sample_0 in fold_a
        assert 15 in lengths  # From sample_1 in fold_a
        assert 12 in lengths  # From sample_2 in fold_b
