"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def sample_input_data():
    """Generate sample input data for testing."""
    # Time series of length 10, dimension 3
    return np.random.randn(10, 3)


@pytest.fixture
def sample_target_data():
    """Generate sample target data for testing."""
    # One-hot encoded targets of length 10, 2 classes
    targets = np.zeros((10, 2))
    targets[:5, 0] = 1.0  # First 5 timesteps: class 0
    targets[5:, 1] = 1.0  # Last 5 timesteps: class 1
    return targets
