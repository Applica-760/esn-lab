"""Tests for data processing utilities."""

import numpy as np
import pytest

from esn_lab.utils.data_processing import make_onehot


class TestMakeOnehot:
    """Test make_onehot function."""

    def test_basic_onehot_encoding(self):
        """Test basic one-hot encoding."""
        result = make_onehot(class_id=1, T=5, num_of_class=3)

        # Check shape
        assert result.shape == (5, 3)

        # Check that all rows are identical
        assert np.all(result == result[0])

        # Check that the one-hot encoding is correct
        expected = np.array([0, 1, 0])
        np.testing.assert_array_equal(result[0], expected)

    def test_onehot_class_0(self):
        """Test one-hot encoding for class 0."""
        result = make_onehot(class_id=0, T=3, num_of_class=4)

        assert result.shape == (3, 4)
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_array_equal(result[0], expected)

    def test_onehot_last_class(self):
        """Test one-hot encoding for the last class."""
        result = make_onehot(class_id=2, T=3, num_of_class=3)

        assert result.shape == (3, 3)
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(result[0], expected)

    def test_onehot_single_timestep(self):
        """Test one-hot encoding with T=1."""
        result = make_onehot(class_id=1, T=1, num_of_class=3)

        assert result.shape == (1, 3)
        expected = np.array([0, 1, 0])
        np.testing.assert_array_equal(result[0], expected)

    def test_onehot_single_class(self):
        """Test one-hot encoding with only one class (edge case)."""
        result = make_onehot(class_id=0, T=5, num_of_class=1)

        assert result.shape == (5, 1)
        expected = np.array([1])
        np.testing.assert_array_equal(result[0], expected)

    def test_onehot_many_timesteps(self):
        """Test one-hot encoding with many timesteps."""
        result = make_onehot(class_id=2, T=1000, num_of_class=5)

        assert result.shape == (1000, 5)

        # Check first and last rows
        expected = np.array([0, 0, 1, 0, 0])
        np.testing.assert_array_equal(result[0], expected)
        np.testing.assert_array_equal(result[-1], expected)

        # Check all rows are identical
        assert np.all(result == result[0])

    def test_onehot_many_classes(self):
        """Test one-hot encoding with many classes."""
        num_classes = 100
        class_id = 42
        result = make_onehot(class_id=class_id, T=10, num_of_class=num_classes)

        assert result.shape == (10, num_classes)

        # Check that only the correct index is 1
        assert result[0, class_id] == 1
        assert np.sum(result[0]) == 1

    def test_onehot_dtype(self):
        """Test that output has correct dtype (should be float)."""
        result = make_onehot(class_id=1, T=5, num_of_class=3)

        # np.eye returns float64 by default
        assert result.dtype == np.float64

    def test_onehot_all_rows_identical(self):
        """Test that all rows are identical (tiling works correctly)."""
        result = make_onehot(class_id=1, T=10, num_of_class=3)

        # Check that all rows are the same
        for i in range(1, len(result)):
            np.testing.assert_array_equal(result[0], result[i])


class TestMakeOnehotEdgeCases:
    """Test edge cases and potential error conditions."""

    def test_invalid_class_id_negative(self):
        """Test with negative class_id (NumPy supports negative indexing).

        Note: NumPy's eye matrix supports negative indexing, so -1 refers
        to the last class. This may not be the intended behavior for this
        function, but it's how the current implementation works.
        """
        # Negative indexing is supported by NumPy, -1 refers to last class
        result = make_onehot(class_id=-1, T=5, num_of_class=3)

        # -1 should give us the last class (index 2)
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(result[0], expected)

    def test_invalid_class_id_too_large(self):
        """Test with class_id >= num_of_class (should raise IndexError)."""
        with pytest.raises(IndexError):
            make_onehot(class_id=3, T=5, num_of_class=3)

        with pytest.raises(IndexError):
            make_onehot(class_id=10, T=5, num_of_class=3)

    def test_zero_timesteps(self):
        """Test with T=0 (should return empty array with correct shape)."""
        result = make_onehot(class_id=1, T=0, num_of_class=3)

        assert result.shape == (0, 3)
        assert len(result) == 0

    def test_zero_classes(self):
        """Test with num_of_class=0 (edge case)."""
        # This will raise an IndexError when trying to index into empty eye matrix
        with pytest.raises(IndexError):
            make_onehot(class_id=0, T=5, num_of_class=0)

    def test_boundary_class_id(self):
        """Test with class_id at the boundary (num_of_class - 1)."""
        num_classes = 10
        class_id = num_classes - 1

        result = make_onehot(class_id=class_id, T=5, num_of_class=num_classes)

        assert result.shape == (5, num_classes)
        assert result[0, class_id] == 1
        assert result[0, 0] == 0

    def test_large_timesteps(self):
        """Test with very large T value."""
        result = make_onehot(class_id=0, T=10000, num_of_class=2)

        assert result.shape == (10000, 2)
        # Verify memory efficiency by checking it's tiled, not individually created
        assert np.all(result[:, 0] == 1)
        assert np.all(result[:, 1] == 0)


class TestMakeOnehotOutputProperties:
    """Test output properties of make_onehot."""

    def test_output_is_binary(self):
        """Test that output contains only 0s and 1s."""
        result = make_onehot(class_id=2, T=10, num_of_class=5)

        unique_values = np.unique(result)
        np.testing.assert_array_equal(unique_values, [0.0, 1.0])

    def test_output_sum_per_row(self):
        """Test that each row sums to 1 (valid one-hot encoding)."""
        result = make_onehot(class_id=3, T=10, num_of_class=5)

        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(10))

    def test_output_is_copy_not_view(self):
        """Test that output can be safely modified without affecting internal state."""
        result1 = make_onehot(class_id=1, T=5, num_of_class=3)
        result2 = make_onehot(class_id=1, T=5, num_of_class=3)

        # Modify result1
        result1[0, 0] = 999

        # result2 should be unaffected
        assert result2[0, 0] == 0


class TestMakeOnehotIntegration:
    """Integration tests for make_onehot with typical use cases."""

    def test_binary_classification(self):
        """Test one-hot encoding for binary classification."""
        # Class 0
        result0 = make_onehot(class_id=0, T=10, num_of_class=2)
        np.testing.assert_array_equal(result0[0], [1, 0])

        # Class 1
        result1 = make_onehot(class_id=1, T=10, num_of_class=2)
        np.testing.assert_array_equal(result1[0], [0, 1])

    def test_multiclass_classification(self):
        """Test one-hot encoding for multi-class classification."""
        num_classes = 10
        T = 20

        for class_id in range(num_classes):
            result = make_onehot(class_id=class_id, T=T, num_of_class=num_classes)

            # Check shape
            assert result.shape == (T, num_classes)

            # Check that only the correct class is 1
            expected = np.zeros(num_classes)
            expected[class_id] = 1
            np.testing.assert_array_equal(result[0], expected)

    def test_variable_length_sequences(self):
        """Test one-hot encoding with different sequence lengths (T values)."""
        class_id = 2
        num_classes = 5

        for T in [1, 5, 10, 50, 100]:
            result = make_onehot(class_id=class_id, T=T, num_of_class=num_classes)

            assert result.shape == (T, num_classes)
            assert np.all(result[:, class_id] == 1)
            assert np.sum(result) == T  # Total sum should equal T

    def test_consistency(self):
        """Test that repeated calls produce identical results."""
        class_id = 3
        T = 15
        num_classes = 7

        result1 = make_onehot(class_id=class_id, T=T, num_of_class=num_classes)
        result2 = make_onehot(class_id=class_id, T=T, num_of_class=num_classes)

        np.testing.assert_array_equal(result1, result2)
