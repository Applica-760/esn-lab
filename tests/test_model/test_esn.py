"""Tests for ESN model components."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose

from esn_lab.model.esn import Input, Reservoir, Output, ESN


class TestInput:
    """Tests for Input layer."""

    def test_initialization(self):
        """Test Input layer initialization."""
        N_u, N_x = 3, 100
        input_scale = 0.1
        seed = 42

        input_layer = Input(N_u, N_x, input_scale, seed=seed)

        # Check weight matrix shape
        assert input_layer.Win.shape == (N_x, N_u)

        # Check weight values are within the specified range
        assert np.all(input_layer.Win >= -input_scale)
        assert np.all(input_layer.Win <= input_scale)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same weights."""
        N_u, N_x = 3, 100
        input_scale = 0.1
        seed = 42

        input_layer1 = Input(N_u, N_x, input_scale, seed=seed)
        input_layer2 = Input(N_u, N_x, input_scale, seed=seed)

        assert_array_equal(input_layer1.Win, input_layer2.Win)

    def test_different_seeds_produce_different_weights(self):
        """Test that different seeds produce different weights."""
        N_u, N_x = 3, 100
        input_scale = 0.1

        input_layer1 = Input(N_u, N_x, input_scale, seed=42)
        input_layer2 = Input(N_u, N_x, input_scale, seed=43)

        assert not np.array_equal(input_layer1.Win, input_layer2.Win)

    def test_forward_pass(self):
        """Test forward pass through Input layer."""
        N_u, N_x = 3, 100
        input_scale = 0.1
        seed = 42

        input_layer = Input(N_u, N_x, input_scale, seed=seed)
        u = np.random.randn(N_u)

        x_in = input_layer(u)

        # Check output shape
        assert x_in.shape == (N_x,)

        # Check that output matches matrix multiplication
        expected = np.dot(input_layer.Win, u)
        assert_array_almost_equal(x_in, expected)

    def test_forward_pass_with_zero_input(self):
        """Test forward pass with zero input."""
        N_u, N_x = 3, 100
        input_scale = 0.1

        input_layer = Input(N_u, N_x, input_scale, seed=42)
        u = np.zeros(N_u)

        x_in = input_layer(u)

        assert_array_almost_equal(x_in, np.zeros(N_x))


class TestReservoir:
    """Tests for Reservoir layer."""

    def test_initialization(self):
        """Test Reservoir initialization."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 0.3
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)

        # Check weight matrix shape
        assert reservoir.W.shape == (N_x, N_x)

        # Check state vector shape and initialization
        assert reservoir.x.shape == (N_x,)
        assert_array_equal(reservoir.x, np.zeros(N_x))

        # Check leaking rate
        assert reservoir.alpha == leaking_rate

        # Check activation function
        assert reservoir.activation_func == activation_func

    def test_spectral_radius_scaling(self):
        """Test that spectral radius is correctly scaled."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 1.0
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)

        # Calculate spectral radius
        eigenvalues = np.linalg.eigvals(reservoir.W)
        spectral_radius = np.max(np.abs(eigenvalues))

        # Check that spectral radius is close to rho (within numerical tolerance)
        assert_allclose(spectral_radius, rho, rtol=1e-10, atol=1e-10)

    def test_sparsity(self):
        """Test that connection matrix has correct sparsity."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 1.0
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)

        # Count non-zero elements
        non_zero_count = np.count_nonzero(reservoir.W)
        total_elements = N_x * N_x
        actual_density = non_zero_count / total_elements

        # Expected number of edges in undirected graph: N_x*(N_x-1)*density/2
        # When converted to directed adjacency matrix, both (i,j) and (j,i) are filled
        # So expected non-zero count is approximately N_x*(N_x-1)*density
        expected_non_zero = int(N_x * (N_x - 1) * density)

        # Allow some tolerance due to graph generation randomness
        assert abs(non_zero_count - expected_non_zero) / expected_non_zero < 0.1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same reservoir weights."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 1.0
        activation_func = np.tanh
        seed = 42

        reservoir1 = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)
        reservoir2 = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)

        assert_array_almost_equal(reservoir1.W, reservoir2.W)

    def test_state_update(self):
        """Test reservoir state update."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 0.3
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)
        x_in = np.random.randn(N_x)

        # Store initial state
        x_prev = reservoir.x.copy()

        # Update state
        x_new = reservoir(x_in)

        # Check output shape
        assert x_new.shape == (N_x,)

        # Verify leaky integration formula
        expected = (1.0 - leaking_rate) * x_prev + \
                   leaking_rate * activation_func(np.dot(reservoir.W, x_prev) + x_in)
        assert_array_almost_equal(x_new, expected)

        # Check that internal state was updated
        assert_array_almost_equal(reservoir.x, x_new)

    def test_state_update_with_full_leaking(self):
        """Test state update with leaking_rate=1.0 (no memory)."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 1.0
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)
        x_in = np.random.randn(N_x)

        x_prev = reservoir.x.copy()
        x_new = reservoir(x_in)

        # With leaking_rate=1.0, previous state should not contribute
        expected = activation_func(np.dot(reservoir.W, x_prev) + x_in)
        assert_array_almost_equal(x_new, expected)

    def test_state_update_multiple_steps(self):
        """Test multiple sequential state updates."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 0.3
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)

        # Apply multiple updates
        for _ in range(10):
            x_in = np.random.randn(N_x)
            reservoir(x_in)

        # Check that state is not all zeros (has been updated)
        assert not np.allclose(reservoir.x, np.zeros(N_x))

    def test_reset_reservoir_state(self):
        """Test reservoir state reset."""
        N_x = 100
        density = 0.1
        rho = 0.9
        leaking_rate = 0.3
        activation_func = np.tanh
        seed = 42

        reservoir = Reservoir(N_x, density, rho, activation_func, leaking_rate, seed=seed)

        # Update state
        x_in = np.random.randn(N_x)
        reservoir(x_in)

        # Verify state is not zero
        assert not np.allclose(reservoir.x, np.zeros(N_x))

        # Reset state
        reservoir.reset_reservoir_state()

        # Verify state is zero
        assert_array_equal(reservoir.x, np.zeros(N_x))


class TestOutput:
    """Tests for Output layer."""

    def test_initialization(self):
        """Test Output layer initialization."""
        N_x, N_y = 100, 5
        seed = 42

        output_layer = Output(N_x, N_y, seed=seed)

        # Check weight matrix shape
        assert output_layer.Wout.shape == (N_y, N_x)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same weights."""
        N_x, N_y = 100, 5
        seed = 42

        output_layer1 = Output(N_x, N_y, seed=seed)
        output_layer2 = Output(N_x, N_y, seed=seed)

        assert_array_almost_equal(output_layer1.Wout, output_layer2.Wout)

    def test_forward_pass(self):
        """Test forward pass through Output layer."""
        N_x, N_y = 100, 5
        seed = 42

        output_layer = Output(N_x, N_y, seed=seed)
        x = np.random.randn(N_x)

        y = output_layer(x)

        # Check output shape
        assert y.shape == (N_y,)

        # Check that output matches matrix multiplication
        expected = np.dot(output_layer.Wout, x)
        assert_array_almost_equal(y, expected)

    def test_setweight(self):
        """Test setting output weights."""
        N_x, N_y = 100, 5
        seed = 42

        output_layer = Output(N_x, N_y, seed=seed)

        # Create new weights
        new_weights = np.random.randn(N_y, N_x)

        # Set weights
        output_layer.setweight(new_weights)

        # Verify weights were updated
        assert_array_equal(output_layer.Wout, new_weights)

    def test_forward_pass_after_setweight(self):
        """Test that forward pass uses updated weights."""
        N_x, N_y = 100, 5
        seed = 42

        output_layer = Output(N_x, N_y, seed=seed)
        x = np.random.randn(N_x)

        # Create and set new weights
        new_weights = np.random.randn(N_y, N_x)
        output_layer.setweight(new_weights)

        # Forward pass
        y = output_layer(x)

        # Verify output uses new weights
        expected = np.dot(new_weights, x)
        assert_array_almost_equal(y, expected)


class TestESN:
    """Tests for complete ESN model."""

    def test_initialization(self):
        """Test ESN initialization."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9
        leaking_rate = 0.3

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho,
                  activation_func=np.tanh, leaking_rate=leaking_rate)

        # Check dimensions
        assert esn.N_u == N_u
        assert esn.N_y == N_y
        assert esn.N_x == N_x

        # Check parameters
        assert esn.density == density
        assert esn.input_scale == input_scale
        assert esn.rho == rho

        # Check layer initialization
        assert isinstance(esn.Input, Input)
        assert isinstance(esn.Reservoir, Reservoir)
        assert isinstance(esn.Output, Output)

        # Check y_prev initialization
        assert esn.y_prev.shape == (N_y,)
        assert_array_equal(esn.y_prev, np.zeros(N_y))

    def test_layer_dimensions_consistency(self):
        """Test that all layers have consistent dimensions."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho)

        # Input layer: Win should be (N_x, N_u)
        assert esn.Input.Win.shape == (N_x, N_u)

        # Reservoir: W should be (N_x, N_x), x should be (N_x,)
        assert esn.Reservoir.W.shape == (N_x, N_x)
        assert esn.Reservoir.x.shape == (N_x,)

        # Output layer: Wout should be (N_y, N_x)
        assert esn.Output.Wout.shape == (N_y, N_x)

    def test_activation_and_output_functions(self):
        """Test custom activation and output functions."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9

        def custom_activation(x):
            return x  # Identity

        def custom_output(x):
            return x * 2

        def custom_inv_output(x):
            return x / 2

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho,
                  activation_func=custom_activation,
                  output_func=custom_output,
                  inv_output_func=custom_inv_output)

        assert esn.Reservoir.activation_func == custom_activation
        assert esn.output_func == custom_output
        assert esn.inv_output_func == custom_inv_output

    def test_different_leaking_rates(self):
        """Test ESN with different leaking rates."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9

        # Test with leaking_rate = 0.3
        esn1 = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=0.3)
        assert esn1.Reservoir.alpha == 0.3

        # Test with leaking_rate = 1.0 (default)
        esn2 = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=1.0)
        assert esn2.Reservoir.alpha == 1.0

    def test_full_forward_pass(self):
        """Test complete forward pass through ESN."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9
        leaking_rate = 0.3

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=leaking_rate)

        # Input vector
        u = np.random.randn(N_u)

        # Manual forward pass
        x_in = esn.Input(u)
        x = esn.Reservoir(x_in)
        y = esn.Output(x)

        # Check shapes
        assert x_in.shape == (N_x,)
        assert x.shape == (N_x,)
        assert y.shape == (N_y,)

    def test_sequential_forward_passes(self):
        """Test multiple sequential forward passes."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9
        leaking_rate = 0.3

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=leaking_rate)

        # Time series of 20 steps
        T = 20
        outputs = []

        for t in range(T):
            u = np.random.randn(N_u)
            x_in = esn.Input(u)
            x = esn.Reservoir(x_in)
            y = esn.Output(x)
            outputs.append(y)

        # Check that we got T outputs
        assert len(outputs) == T

        # Check all outputs have correct shape
        for y in outputs:
            assert y.shape == (N_y,)

        # Check that reservoir state was updated (not all zeros)
        assert not np.allclose(esn.Reservoir.x, np.zeros(N_x))

    def test_reservoir_state_accumulation(self):
        """Test that reservoir state accumulates information over time."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9
        leaking_rate = 0.3

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=leaking_rate)

        # Initial state
        state_initial = esn.Reservoir.x.copy()

        # First input
        u1 = np.random.randn(N_u)
        x_in1 = esn.Input(u1)
        esn.Reservoir(x_in1)
        state_after_1 = esn.Reservoir.x.copy()

        # Second input
        u2 = np.random.randn(N_u)
        x_in2 = esn.Input(u2)
        esn.Reservoir(x_in2)
        state_after_2 = esn.Reservoir.x.copy()

        # States should be different at each step
        assert not np.allclose(state_initial, state_after_1)
        assert not np.allclose(state_after_1, state_after_2)

    def test_output_weight_update(self):
        """Test updating output weights after training."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho)

        # Initial output
        u = np.random.randn(N_u)
        x_in = esn.Input(u)
        x = esn.Reservoir(x_in)
        y_before = esn.Output(x)

        # Update output weights (simulate training)
        new_Wout = np.random.randn(N_y, N_x)
        esn.Output.setweight(new_Wout)

        # Reset reservoir and run again with same input
        esn.Reservoir.reset_reservoir_state()
        x_in = esn.Input(u)
        x = esn.Reservoir(x_in)
        y_after = esn.Output(x)

        # Outputs should be different
        assert not np.allclose(y_before, y_after)

    def test_reservoir_reset(self):
        """Test that reservoir reset works correctly in ESN context."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9
        leaking_rate = 0.3

        esn = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=leaking_rate)

        # Run some steps to accumulate state
        for _ in range(10):
            u = np.random.randn(N_u)
            x_in = esn.Input(u)
            esn.Reservoir(x_in)

        # Verify state is not zero
        assert not np.allclose(esn.Reservoir.x, np.zeros(N_x))

        # Reset
        esn.Reservoir.reset_reservoir_state()

        # Verify state is zero
        assert_array_equal(esn.Reservoir.x, np.zeros(N_x))

    def test_deterministic_behavior_with_fixed_input(self):
        """Test that ESN produces same output for same input sequence."""
        N_u, N_y, N_x = 3, 5, 100
        density = 0.1
        input_scale = 0.1
        rho = 0.9
        leaking_rate = 0.3
        seed = 42

        # Create two identical ESNs
        np.random.seed(seed)
        esn1 = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=leaking_rate)

        np.random.seed(seed)
        esn2 = ESN(N_u, N_y, N_x, density, input_scale, rho, leaking_rate=leaking_rate)

        # Same input sequence
        np.random.seed(seed)
        input_sequence = [np.random.randn(N_u) for _ in range(10)]

        # Run both ESNs
        outputs1 = []
        outputs2 = []

        for u in input_sequence:
            x_in1 = esn1.Input(u)
            x1 = esn1.Reservoir(x_in1)
            y1 = esn1.Output(x1)
            outputs1.append(y1)

            x_in2 = esn2.Input(u)
            x2 = esn2.Reservoir(x_in2)
            y2 = esn2.Output(x2)
            outputs2.append(y2)

        # Outputs should be identical
        for y1, y2 in zip(outputs1, outputs2):
            assert_array_almost_equal(y1, y2)
