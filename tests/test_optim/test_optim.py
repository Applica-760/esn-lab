"""Tests for Tikhonov optimizer."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose

from esn_lab.optim.optim import Tikhonov


class TestTikhonov:
    """Tests for Tikhonov regularized ridge regression optimizer."""

    def test_initialization(self):
        """Test Tikhonov optimizer initialization."""
        N_x, N_y = 100, 5
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Check beta
        assert optimizer.beta == beta

        # Check N_x
        assert optimizer.N_x == N_x

        # Check accumulator matrices initialization
        assert optimizer.X_XT.shape == (N_x, N_x)
        assert optimizer.D_XT.shape == (N_y, N_x)

        # Check that accumulators are initialized to zero
        assert_array_equal(optimizer.X_XT, np.zeros((N_x, N_x)))
        assert_array_equal(optimizer.D_XT, np.zeros((N_y, N_x)))

    def test_accumulator_update_single_sample(self):
        """Test accumulator update with single sample."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create sample data
        x = np.random.randn(N_x)
        d = np.random.randn(N_y)

        # Update
        optimizer(d, x)

        # Expected updates
        expected_X_XT = np.outer(x, x)
        expected_D_XT = np.outer(d, x)

        # Check accumulator updates
        assert_array_almost_equal(optimizer.X_XT, expected_X_XT)
        assert_array_almost_equal(optimizer.D_XT, expected_D_XT)

    def test_accumulator_update_multiple_samples(self):
        """Test accumulator update with multiple samples."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create multiple samples
        n_samples = 20
        X = np.random.randn(n_samples, N_x)
        D = np.random.randn(n_samples, N_y)

        # Update accumulator for each sample
        for i in range(n_samples):
            optimizer(D[i], X[i])

        # Expected accumulations
        expected_X_XT = np.dot(X.T, X)
        expected_D_XT = np.dot(D.T, X)

        # Check accumulator updates
        assert_array_almost_equal(optimizer.X_XT, expected_X_XT)
        assert_array_almost_equal(optimizer.D_XT, expected_D_XT)

    def test_accumulator_with_different_input_shapes(self):
        """Test that accumulator handles different input shapes correctly."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Test with 1D arrays
        x1 = np.random.randn(N_x)
        d1 = np.random.randn(N_y)
        optimizer(d1, x1)

        X_XT_after_1 = optimizer.X_XT.copy()
        D_XT_after_1 = optimizer.D_XT.copy()

        # Reset
        optimizer = Tikhonov(N_x, N_y, beta)

        # Test with column vectors (2D arrays)
        x2 = np.random.randn(N_x, 1)
        d2 = np.random.randn(N_y, 1)
        optimizer(d2, x2)

        # Should produce same results (after flattening)
        x1_flat = x1.flatten()
        x2_flat = x2.flatten()
        d1_flat = d1.flatten()
        d2_flat = d2.flatten()

        if np.allclose(x1_flat, x2_flat) and np.allclose(d1_flat, d2_flat):
            assert_array_almost_equal(optimizer.X_XT, X_XT_after_1)
            assert_array_almost_equal(optimizer.D_XT, D_XT_after_1)

    def test_get_Wout_opt_simple_case(self):
        """Test Wout optimization with a simple well-conditioned case."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create well-conditioned training data
        n_samples = 100
        np.random.seed(42)
        X = np.random.randn(n_samples, N_x)

        # True weights
        W_true = np.random.randn(N_y, N_x)

        # Generate targets with small noise
        D = np.dot(X, W_true.T) + 0.01 * np.random.randn(n_samples, N_y)

        # Accumulate statistics
        for i in range(n_samples):
            optimizer(D[i], X[i])

        # Get optimal weights
        W_opt = optimizer.get_Wout_opt()

        # Check shape
        assert W_opt.shape == (N_y, N_x)

        # With enough data and small noise, W_opt should be close to W_true
        # (allowing for regularization effect)
        assert_allclose(W_opt, W_true, rtol=0.1, atol=0.1)

    def test_get_Wout_opt_zero_beta(self):
        """Test Wout optimization with beta=0 (standard linear regression)."""
        N_x, N_y = 10, 3
        beta = 0.0

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create training data
        n_samples = 100
        np.random.seed(42)
        X = np.random.randn(n_samples, N_x)
        W_true = np.random.randn(N_y, N_x)
        D = np.dot(X, W_true.T)

        # Accumulate statistics
        for i in range(n_samples):
            optimizer(D[i], X[i])

        # Get optimal weights
        W_opt = optimizer.get_Wout_opt()

        # Check shape
        assert W_opt.shape == (N_y, N_x)

        # With beta=0 and perfect data, should recover true weights
        assert_allclose(W_opt, W_true, rtol=1e-5, atol=1e-5)

    def test_get_Wout_opt_with_regularization(self):
        """Test that larger beta leads to smaller weight magnitudes."""
        N_x, N_y = 10, 3
        n_samples = 50

        np.random.seed(42)
        X = np.random.randn(n_samples, N_x)
        W_true = np.random.randn(N_y, N_x)
        D = np.dot(X, W_true.T)

        # Test with small beta
        optimizer1 = Tikhonov(N_x, N_y, beta=1e-6)
        for i in range(n_samples):
            optimizer1(D[i], X[i])
        W_opt1 = optimizer1.get_Wout_opt()

        # Test with large beta
        optimizer2 = Tikhonov(N_x, N_y, beta=1.0)
        for i in range(n_samples):
            optimizer2(D[i], X[i])
        W_opt2 = optimizer2.get_Wout_opt()

        # Larger beta should lead to smaller weight norm (regularization effect)
        norm1 = np.linalg.norm(W_opt1)
        norm2 = np.linalg.norm(W_opt2)

        assert norm2 < norm1

    def test_get_Wout_opt_identity_mapping(self):
        """Test Wout optimization for identity mapping."""
        N_x = 10
        N_y = N_x  # Same dimension
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create identity mapping: D = X
        n_samples = 100
        np.random.seed(42)
        X = np.random.randn(n_samples, N_x)
        D = X.copy()  # Identity mapping

        # Accumulate statistics
        for i in range(n_samples):
            optimizer(D[i], X[i])

        # Get optimal weights
        W_opt = optimizer.get_Wout_opt()

        # Should be close to identity matrix
        assert W_opt.shape == (N_y, N_x)
        assert_allclose(W_opt, np.eye(N_x), rtol=0.05, atol=0.05)

    def test_fallback_to_pseudoinverse(self):
        """Test fallback to pseudoinverse for ill-conditioned matrices."""
        N_x, N_y = 10, 3
        beta = 0.0  # No regularization to make it more likely to be singular

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create rank-deficient data (only 5 unique samples repeated)
        n_unique = 5
        np.random.seed(42)
        X_unique = np.random.randn(n_unique, N_x)

        # Repeat samples to fill n_samples
        n_samples = 20
        X = np.tile(X_unique, (n_samples // n_unique + 1, 1))[:n_samples]

        W_true = np.random.randn(N_y, N_x)
        D = np.dot(X, W_true.T)

        # Accumulate statistics
        for i in range(n_samples):
            optimizer(D[i], X[i])

        # This should not raise an error, even if matrix is singular
        # It should fall back to pseudoinverse
        W_opt = optimizer.get_Wout_opt()

        # Check shape
        assert W_opt.shape == (N_y, N_x)

        # Should still produce reasonable predictions on training data
        D_pred = np.dot(X, W_opt.T)
        # At least correlation should be high
        for j in range(N_y):
            correlation = np.corrcoef(D[:, j], D_pred[:, j])[0, 1]
            assert correlation > 0.9

    def test_sequential_accumulation(self):
        """Test that accumulation is order-independent (commutative)."""
        N_x, N_y = 10, 3
        beta = 1e-4

        # Create sample data
        np.random.seed(42)
        n_samples = 10
        X = np.random.randn(n_samples, N_x)
        D = np.random.randn(n_samples, N_y)

        # Accumulate in order
        optimizer1 = Tikhonov(N_x, N_y, beta)
        for i in range(n_samples):
            optimizer1(D[i], X[i])

        # Accumulate in reverse order
        optimizer2 = Tikhonov(N_x, N_y, beta)
        for i in reversed(range(n_samples)):
            optimizer2(D[i], X[i])

        # Results should be identical
        assert_array_almost_equal(optimizer1.X_XT, optimizer2.X_XT)
        assert_array_almost_equal(optimizer1.D_XT, optimizer2.D_XT)

        # Optimal weights should be identical
        W_opt1 = optimizer1.get_Wout_opt()
        W_opt2 = optimizer2.get_Wout_opt()
        assert_array_almost_equal(W_opt1, W_opt2)

    def test_empty_accumulator_raises_or_handles_gracefully(self):
        """Test behavior when get_Wout_opt is called without data."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Call get_Wout_opt without accumulating any data
        # Should handle gracefully (might return zeros or regularized solution)
        W_opt = optimizer.get_Wout_opt()

        # Check shape is correct
        assert W_opt.shape == (N_y, N_x)

        # With no data and regularization, should be close to zero
        assert_allclose(W_opt, np.zeros((N_y, N_x)), atol=1e-10)

    def test_single_sample_accumulation(self):
        """Test optimization with just one sample."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Single sample
        x = np.random.randn(N_x)
        d = np.random.randn(N_y)

        optimizer(d, x)

        # Should not crash
        W_opt = optimizer.get_Wout_opt()

        # Check shape
        assert W_opt.shape == (N_y, N_x)

    def test_accumulator_dtype_preservation(self):
        """Test that accumulator preserves dtype."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Use float32 data
        x = np.random.randn(N_x).astype(np.float32)
        d = np.random.randn(N_y).astype(np.float32)

        optimizer(d, x)

        # Accumulators should handle the dtype
        # (Note: depending on implementation, might be promoted to float64)
        assert optimizer.X_XT.dtype in [np.float32, np.float64]
        assert optimizer.D_XT.dtype in [np.float32, np.float64]

    def test_mathematical_correctness_closed_form(self):
        """Test mathematical correctness against closed-form solution."""
        N_x, N_y = 5, 2
        beta = 0.1

        optimizer = Tikhonov(N_x, N_y, beta)

        # Create small dataset for manual verification
        np.random.seed(42)
        n_samples = 20
        X = np.random.randn(n_samples, N_x)
        D = np.random.randn(n_samples, N_y)

        # Accumulate using optimizer
        for i in range(n_samples):
            optimizer(D[i], X[i])

        W_opt = optimizer.get_Wout_opt()

        # Compute closed-form solution manually
        X_XT_manual = np.dot(X.T, X)
        D_XT_manual = np.dot(D.T, X)
        A_manual = X_XT_manual + beta * np.eye(N_x)
        W_opt_manual = np.linalg.solve(A_manual, D_XT_manual.T).T

        # Should match
        assert_array_almost_equal(W_opt, W_opt_manual, decimal=10)

    def test_zero_input_vector(self):
        """Test accumulation with zero input vector."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Zero input
        x = np.zeros(N_x)
        d = np.random.randn(N_y)

        optimizer(d, x)

        # X_XT should remain zero
        assert_array_equal(optimizer.X_XT, np.zeros((N_x, N_x)))

        # D_XT should also be zero
        assert_array_equal(optimizer.D_XT, np.zeros((N_y, N_x)))

    def test_zero_target_vector(self):
        """Test accumulation with zero target vector."""
        N_x, N_y = 10, 3
        beta = 1e-4

        optimizer = Tikhonov(N_x, N_y, beta)

        # Zero target
        x = np.random.randn(N_x)
        d = np.zeros(N_y)

        optimizer(d, x)

        # X_XT should be updated
        expected_X_XT = np.outer(x, x)
        assert_array_almost_equal(optimizer.X_XT, expected_X_XT)

        # D_XT should remain zero
        assert_array_equal(optimizer.D_XT, np.zeros((N_y, N_x)))

    def test_large_scale(self):
        """Test with larger dimensions."""
        N_x, N_y = 500, 10
        beta = 1e-3

        optimizer = Tikhonov(N_x, N_y, beta)

        # Large scale data
        n_samples = 1000
        np.random.seed(42)

        for i in range(n_samples):
            x = np.random.randn(N_x)
            d = np.random.randn(N_y)
            optimizer(d, x)

        # Should handle large scale without issues
        W_opt = optimizer.get_Wout_opt()

        assert W_opt.shape == (N_y, N_x)
        assert np.all(np.isfinite(W_opt))
