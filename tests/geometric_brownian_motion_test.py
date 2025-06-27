import numpy as np
import pytest

from source.processes.geometric_brownian_motion import GeometricBrownianMotion


class TestGeometricBrownianMotion:
    # Initialization Tests
    def test_initialization_default_params(self):
        """Test initialization with default parameters"""
        gbm = GeometricBrownianMotion()
        assert gbm.drift == 0
        assert gbm.diffusion == 1
        assert gbm.dimension == 1
        assert gbm.initial_value == 1

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        gbm = GeometricBrownianMotion(
            drift=0.5, diffusion=2.0, initial_value=5, dimension=3
        )
        assert gbm.drift == 0.5
        assert gbm.diffusion == 2.0
        assert gbm.dimension == 3
        assert gbm.initial_value == 5

    # Discretization Method Test
    def test_discretization_method(self):
        """Test the _discretize method produces expected output"""
        gbm = GeometricBrownianMotion(drift=0.1, diffusion=0.5)
        prev_value = 1.0
        dt = 0.1
        random_normal = np.array([0.5, -0.5])

        result = gbm._discretize(prev_value, dt, random_normal)
        expected = prev_value * np.exp((0.1 - 0.5 / 2) * dt + np.sqrt(0.5 * dt) * random_normal)
        np.testing.assert_allclose(result, expected)

    # Simulation Tests
    def test_simulation_shape(self):
        """Test simulation output has correct shape"""
        gbm = GeometricBrownianMotion()
        n_paths = 100
        n_steps = 50
        paths = gbm.simulate(n_paths, n_steps, dt=0.1)
        assert paths.shape == (n_paths, n_steps + 1)

    def test_simulation_initial_value(self):
        """Test all paths start at initial value"""
        gbm = GeometricBrownianMotion()
        paths = gbm.simulate(10, 5, dt=0.1)
        assert np.all(paths[:, 0] == gbm.initial_value)

    def test_simulation_with_seed(self):
        """Test reproducibility with seed"""
        gbm = GeometricBrownianMotion()
        paths1 = gbm.simulate(10, 5, dt=0.1, seed=42)
        paths2 = gbm.simulate(10, 5, dt=0.1, seed=42)
        np.testing.assert_array_equal(paths1, paths2)

    def test_simulation_with_times(self):
        """Test return_times option works correctly"""
        gbm = GeometricBrownianMotion()
        n_steps = 10
        dt = 0.1
        paths, times = gbm.simulate(5, n_steps, dt, return_times=True)
        expected_times = np.linspace(0, n_steps * dt, n_steps + 1)
        np.testing.assert_allclose(times, expected_times)

    # Statistical properties Tests
    def test_mean_calculation(self):
        """Test mean calculation is correct"""
        gbm = GeometricBrownianMotion(drift=0.2)
        t_values = np.array([0, 1, 2, 5])
        expected_means = np.exp(0.2 * t_values)
        np.testing.assert_allclose(gbm.mean(t_values), expected_means)

    def test_variance_calculation(self):
        """Test variance calculation is correct"""
        gbm = GeometricBrownianMotion(diffusion=0.5)
        t_values = np.array([0, 1, 2, 5])
        expected_variances = np.exp(0.5 * t_values) - 1
        np.testing.assert_allclose(gbm.variance(t_values), expected_variances)

    def test_skewness(self):
        """Test skewness calculation is correct"""
        gbm = GeometricBrownianMotion(diffusion=0.5)
        t_values = np.array([0, 1, 2, 5])
        expected_skewness = (np.exp(0.5 * t_values) + 2) * np.sqrt(
            np.exp(0.5 * t_values) - 1
        )
        np.testing.assert_allclose(gbm.skewness(t_values), expected_skewness)

    def test_kurtosis(self):
        """Test kurtosis calculation is correct"""
        gbm = GeometricBrownianMotion(diffusion=0.5)
        t_values = np.array([0, 1, 2, 5])
        expected_kurtosis = (
            np.exp(4 * 0.5 * t_values)
            + 2 * np.exp(3 * 0.5 * t_values)
            + 3 * np.exp(2 * 0.5 * t_values)
            - 3
        )
        np.testing.assert_allclose(gbm.kurtosis(t_values), expected_kurtosis)

    # Parameter validation Tests
    def test_negative_diffusion_raises_error(self):
        """Test that negative diffusion coefficient raises error"""
        with pytest.raises(ValueError):
            GeometricBrownianMotion(diffusion=-1)

    def test_zero_diffusion_raises_error(self):
        """Test that zero diffusion coefficient raises error"""
        with pytest.raises(ValueError):
            GeometricBrownianMotion(diffusion=0)

    def test_negative_dimension_raises_error(self):
        """Test that negative dimension raises error"""
        with pytest.raises(ValueError):
            GeometricBrownianMotion(dimension=-1)

    # Monte Carlo Convergence Tests
    def test_mc_mean_convergence(self):
        """Test simulated paths converge to theoretical mean"""
        drift = 0.1
        gbm = GeometricBrownianMotion(drift=drift)
        n_paths = 10000
        n_steps = 100
        dt = 0.01
        T = n_steps * dt
        paths = gbm.simulate(n_paths, n_steps, dt)

        # Check final time mean
        final_values = paths[:, -1]
        theoretical_mean = gbm.mean(T)
        empirical_mean = np.mean(final_values)

        # Allow 3 standard deviations tolerance
        empirical_std = np.std(final_values, ddof=1) / np.sqrt(n_paths)
        assert abs(empirical_mean - theoretical_mean) < 3 * empirical_std

    def test_mc_variance_convergence(self):
        """Test simulated paths converge to theoretical variance"""
        diffusion = 0.5
        gbm = GeometricBrownianMotion(diffusion=diffusion)
        n_paths = 10000
        n_steps = 100
        dt = 0.01
        T = n_steps * dt
        paths = gbm.simulate(n_paths, n_steps, dt)

        # Check final time variance
        final_values = paths[:, -1]
        theoretical_var = gbm.variance(T)
        theoretical_kurtosis = gbm.kurtosis(T)
        empirical_var = np.var(final_values, ddof=1)

        # Variance of sample variance for lognormal distirbution
        var_of_var = (
            theoretical_kurtosis * theoretical_var**2
            - theoretical_var**2 * (n_paths - 3) / (n_paths - 1)
        ) / n_paths
        std_of_var = np.sqrt(var_of_var)
        assert abs(empirical_var - theoretical_var) < 3 * std_of_var
