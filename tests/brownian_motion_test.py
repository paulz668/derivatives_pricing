import numpy as np
import pytest

from source.processes.brownian_motion import BrownianMotion


class TestBrownianMotion:
    # Initialization Tests
    def test_initialization_default_params(self):
        """Test initialization with default parameters"""
        bm = BrownianMotion()
        assert bm.drift == 0
        assert bm.diffusion == 1
        assert bm.dimension == 1
        assert bm.initial_value == 0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        bm = BrownianMotion(drift=0.5, diffusion=2.0, dimension=3)
        assert bm.drift == 0.5
        assert bm.diffusion == 2.0
        assert bm.dimension == 3
        assert bm.initial_value == 0

    # Discretization Method Test
    def test_discretization_method(self):
        """Test the _discretize method produces expected output"""
        bm = BrownianMotion(drift=0.1, diffusion=0.5)
        prev_value = 1.0
        dt = 0.1
        random_normal = np.array([0.5, -0.5])

        result = bm._discretize(prev_value, dt, random_normal)
        expected = prev_value + 0.1 * 0.1 + np.sqrt(0.5 * 0.1) * random_normal
        np.testing.assert_allclose(result, expected)

    # Simulation Tests
    def test_simulation_shape(self):
        """Test simulation output has correct shape"""
        bm = BrownianMotion()
        n_paths = 100
        n_steps = 50
        paths = bm.simulate(n_paths, n_steps, dt=0.1)
        assert paths.shape == (n_paths, n_steps + 1)

    def test_simulation_initial_value(self):
        """Test all paths start at initial value"""
        bm = BrownianMotion()
        paths = bm.simulate(10, 5, dt=0.1)
        assert np.all(paths[:, 0] == bm.initial_value)

    def test_simulation_with_seed(self):
        """Test reproducibility with seed"""
        bm = BrownianMotion()
        paths1 = bm.simulate(10, 5, dt=0.1, seed=42)
        paths2 = bm.simulate(10, 5, dt=0.1, seed=42)
        np.testing.assert_array_equal(paths1, paths2)

    def test_simulation_with_times(self):
        """Test return_times option works correctly"""
        bm = BrownianMotion()
        n_steps = 10
        dt = 0.1
        paths, times = bm.simulate(5, n_steps, dt, return_times=True)
        expected_times = np.linspace(0, n_steps * dt, n_steps + 1)
        np.testing.assert_allclose(times, expected_times)

    # Statistical properties Tests
    def test_mean_calculation(self):
        """Test mean calculation is correct"""
        bm = BrownianMotion(drift=0.2)
        t_values = np.array([0, 1, 2, 5])
        expected_means = 0.2 * t_values
        np.testing.assert_allclose(bm.mean(t_values), expected_means)

    def test_variance_calculation(self):
        """Test variance calculation is correct"""
        bm = BrownianMotion(diffusion=0.5)
        t_values = np.array([0, 1, 2, 5])
        expected_variances = 0.5 * t_values
        np.testing.assert_allclose(bm.variance(t_values), expected_variances)

    def test_skewness(self):
        """Test skewness is always zero"""
        bm = BrownianMotion()
        assert bm.skewness() == 0

    def test_kurtosis(self):
        """Test kurtosis is always 3 (normal distribution)"""
        bm = BrownianMotion()
        assert bm.kurtosis() == 3

    # Parameter validation Tests
    def test_negative_diffusion_raises_error(self):
        """Test that negative diffusion coefficient raises error"""
        with pytest.raises(ValueError):
            BrownianMotion(diffusion=-1)

    def test_zero_diffusion_raises_error(self):
        """Test that zero diffusion coefficient raises error"""
        with pytest.raises(ValueError):
            BrownianMotion(diffusion=0)

    def test_negative_dimension_raises_error(self):
        """Test that negative dimension raises error"""
        with pytest.raises(ValueError):
            BrownianMotion(dimension=-1)

    # Monte Carlo Convergence Tests
    def test_mc_mean_convergence(self):
        """Test simulated paths converge to theoretical mean"""
        drift = 0.1
        bm = BrownianMotion(drift=drift)
        n_paths = 10000
        n_steps = 100
        dt = 0.01
        T = n_steps * dt
        paths = bm.simulate(n_paths, n_steps, dt)

        # Check final time mean
        final_values = paths[:, -1]
        theoretical_mean = bm.mean(T)
        empirical_mean = np.mean(final_values)

        # Allow 3 standard deviations tolerance
        empirical_std = np.std(final_values) / np.sqrt(n_paths)
        assert abs(empirical_mean - theoretical_mean) < 3 * empirical_std

    def test_mc_variance_convergence(self):
        """Test simulated paths converge to theoretical variance"""
        diffusion = 0.5
        bm = BrownianMotion(diffusion=diffusion)
        n_paths = 10000
        n_steps = 100
        dt = 0.01
        T = n_steps * dt
        paths = bm.simulate(n_paths, n_steps, dt)

        # Check final time variance
        final_values = paths[:, -1]
        theoretical_var = bm.variance(T)
        empirical_var = np.var(final_values)

        # Variance of sample variance is 2σ⁴/(n-1) for normal distribution
        var_of_var = 2 * (theoretical_var**2) / (n_paths - 1)
        std_of_var = np.sqrt(var_of_var)
        assert abs(empirical_var - theoretical_var) < 3 * std_of_var
