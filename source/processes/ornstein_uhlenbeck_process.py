import numpy as np
from . import stochastic_process as sp


class OrnsteinUhlenbeckProcess(sp.StochasticProcess):
    def __init__(
        self,
        k: np.floating,
        zeta: np.floating,
        sigma: np.floating,
        initial_value=1,
        dimension=1,
    ):
        """
        Parameters:
        -----------
        k: mean reversion rate
        zeta: mean reversion level
        sigma: absolute volatility
        """
        self.k = k
        self.zeta = zeta
        self.sigma = sigma
        self.initial_value = initial_value
        self.dimension = dimension
        self._validate_parameters

    def _discretize(self, prev_value, dt, random_normal):
        return (
            1
            + np.exp(-self.k * dt) * (prev_value - 1)
            + np.sqrt(self.sigma**2 / (2 * self.k) * (1 - np.exp(-2 * self.k * dt)))
            * random_normal
        )

    def _validate_parameters(self):
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.dimension < 1:
            raise ValueError("Dimension must be ≥ 1")

    def simulate(self, n_paths, n_steps, dt, return_times=False, seed=None):
        """Simulate paths of an Ornstein Uhlenbeck process"""

        rng = np.random.default_rng(seed=seed)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.initial_value

        normal_sample = rng.standard_normal((n_paths, n_steps))

        for t in range(1, n_steps + 1):
            paths[:, t] = self._discretize(paths[:, t - 1], dt, normal_sample[:, t - 1])

        if return_times:
            times = np.linspace(0, n_steps * dt, n_steps + 1)
            return paths, times

        return paths

    def mean(self, t):
        return self.zeta + (self.initial_value - self.zeta) * np.exp(-self.k * t)

    def variance(self, t):
        return self.sigma**2 / (2 * self.k) * (1 - np.exp(-2 * self.k * t))

    def skewness(self, t):
        raise NotImplementedError

    def kurtosis(self, t):
        raise NotImplementedError
