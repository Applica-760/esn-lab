"""Tests for create_data_loader factory function."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from esn_lab.pipeline.data.csv_loader import CSVDataLoader
from esn_lab.pipeline.data.factory import create_data_loader
from esn_lab.pipeline.data.npy_loader import NPYDataLoader


@pytest.fixture
def temp_npy_dir():
    """Create a temporary NPY directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create metadata.json
        metadata = {"num_classes": 2}
        with open(tmpdir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Create a fold file
        fold_data = {
            "num_samples": 1,
            "sample_0_id": "sample",
            "sample_0_data": np.zeros((5, 3)),
            "sample_0_class": 0,
        }
        np.savez(tmpdir / "fold_a.npz", **fold_data)

        yield tmpdir


@pytest.fixture
def temp_csv_dir():
    """Create a temporary CSV directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        img_dir.mkdir()

        # Create test image
        img = np.random.randint(0, 256, (5, 10), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "sample.png"), img)

        # Create CSV
        df = pd.DataFrame({
            "file_path": [str(img_dir / "sample.png")],
            "behavior": [0],
        })
        df.to_csv(tmpdir / "10fold_a.csv", index=False)

        yield tmpdir


class TestCreateDataLoaderNPY:
    """Test create_data_loader with NPY configuration."""

    def test_create_npy_loader(self, temp_npy_dir):
        """Test creating NPYDataLoader with type='npy'."""
        config = {
            "type": "npy",
            "npy_dir": str(temp_npy_dir),
        }
        loader = create_data_loader(config)

        assert isinstance(loader, NPYDataLoader)
        assert loader.npy_dir == temp_npy_dir

    def test_npy_loader_missing_npy_dir(self):
        """Test error when npy_dir is not specified for type='npy'."""
        config = {
            "type": "npy",
            # npy_dir is missing
        }

        with pytest.raises(ValueError, match="npy_dir is not specified"):
            create_data_loader(config)

    def test_npy_loader_nonexistent_directory(self):
        """Test error when npy_dir doesn't exist."""
        config = {
            "type": "npy",
            "npy_dir": "/nonexistent/path",
        }

        with pytest.raises(FileNotFoundError):
            create_data_loader(config)


class TestCreateDataLoaderCSV:
    """Test create_data_loader with CSV configuration."""

    def test_create_csv_loader(self, temp_csv_dir):
        """Test creating CSVDataLoader with type='csv'."""
        config = {
            "type": "csv",
            "csv_dir": str(temp_csv_dir),
        }
        loader = create_data_loader(config)

        assert isinstance(loader, CSVDataLoader)
        assert loader.csv_dir == temp_csv_dir

    def test_csv_loader_default_type(self, temp_csv_dir):
        """Test that type='csv' is default when not specified."""
        config = {
            "csv_dir": str(temp_csv_dir),
            # type is not specified
        }
        loader = create_data_loader(config)

        assert isinstance(loader, CSVDataLoader)

    def test_csv_loader_with_fallback(self, temp_csv_dir):
        """Test CSV loader with fallback_csv_dir when csv_dir is not in config."""
        config = {
            "type": "csv",
            # csv_dir is missing
        }
        loader = create_data_loader(config, fallback_csv_dir=str(temp_csv_dir))

        assert isinstance(loader, CSVDataLoader)
        assert loader.csv_dir == temp_csv_dir

    def test_csv_loader_missing_csv_dir(self):
        """Test error when csv_dir and fallback_csv_dir are both missing."""
        config = {
            "type": "csv",
            # csv_dir is missing
        }

        with pytest.raises(ValueError, match="csv_dir is not specified"):
            create_data_loader(config)  # No fallback

    def test_csv_loader_nonexistent_directory(self):
        """Test error when csv_dir doesn't exist."""
        config = {
            "type": "csv",
            "csv_dir": "/nonexistent/path",
        }

        with pytest.raises(FileNotFoundError):
            create_data_loader(config)


class TestCreateDataLoaderBackwardCompatibility:
    """Test backward compatibility features."""

    def test_none_config_with_fallback(self, temp_csv_dir):
        """Test legacy mode: None config with fallback_csv_dir."""
        loader = create_data_loader(None, fallback_csv_dir=str(temp_csv_dir))

        assert isinstance(loader, CSVDataLoader)
        assert loader.csv_dir == temp_csv_dir

    def test_none_config_without_fallback(self):
        """Test error when both config and fallback are None."""
        with pytest.raises(ValueError, match="Either data_source_cfg or fallback_csv_dir"):
            create_data_loader(None, fallback_csv_dir=None)

    def test_csv_dir_preference(self, temp_csv_dir):
        """Test that csv_dir in config takes precedence over fallback."""
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir2 = Path(tmpdir2)

            # Create second CSV directory
            df = pd.DataFrame({
                "file_path": ["dummy.png"],
                "behavior": [0],
            })
            df.to_csv(tmpdir2 / "10fold_a.csv", index=False)

            config = {
                "type": "csv",
                "csv_dir": str(tmpdir2),
            }
            loader = create_data_loader(config, fallback_csv_dir=str(temp_csv_dir))

            # Should use csv_dir from config, not fallback
            assert loader.csv_dir == tmpdir2


class TestCreateDataLoaderInvalidConfig:
    """Test error handling for invalid configurations."""

    def test_unknown_type(self):
        """Test error with unknown data source type."""
        config = {
            "type": "unknown_type",
        }

        with pytest.raises(ValueError, match="Unknown data_source.type"):
            create_data_loader(config)

    def test_invalid_config_structure(self):
        """Test handling of malformed config."""
        config = "not a dict"

        with pytest.raises(AttributeError):
            create_data_loader(config)


class TestCreateDataLoaderSelection:
    """Test that correct loader is selected based on configuration."""

    def test_npy_over_csv_when_both_specified(self, temp_npy_dir, temp_csv_dir):
        """Test that type='npy' creates NPY loader even if csv_dir exists."""
        config = {
            "type": "npy",
            "npy_dir": str(temp_npy_dir),
            "csv_dir": str(temp_csv_dir),  # This should be ignored
        }
        loader = create_data_loader(config)

        assert isinstance(loader, NPYDataLoader)
        assert not isinstance(loader, CSVDataLoader)

    def test_csv_over_npy_when_csv_specified(self, temp_npy_dir, temp_csv_dir):
        """Test that type='csv' creates CSV loader even if npy_dir exists."""
        config = {
            "type": "csv",
            "csv_dir": str(temp_csv_dir),
            "npy_dir": str(temp_npy_dir),  # This should be ignored
        }
        loader = create_data_loader(config)

        assert isinstance(loader, CSVDataLoader)
        assert not isinstance(loader, NPYDataLoader)


class TestCreateDataLoaderPathHandling:
    """Test path handling in create_data_loader."""

    def test_string_path(self, temp_npy_dir):
        """Test that string paths are accepted."""
        config = {
            "type": "npy",
            "npy_dir": str(temp_npy_dir),
        }
        loader = create_data_loader(config)
        assert isinstance(loader, NPYDataLoader)

    def test_path_object(self, temp_npy_dir):
        """Test that Path objects are accepted."""
        config = {
            "type": "npy",
            "npy_dir": temp_npy_dir,  # Path object
        }
        loader = create_data_loader(config)
        assert isinstance(loader, NPYDataLoader)

    def test_relative_path(self, temp_npy_dir):
        """Test handling of relative paths."""
        # Create a relative path
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(temp_npy_dir.parent)
            relative_path = temp_npy_dir.name

            config = {
                "type": "npy",
                "npy_dir": relative_path,
            }
            loader = create_data_loader(config)
            assert isinstance(loader, NPYDataLoader)
        finally:
            os.chdir(original_dir)
