import numpy as np
from . import stochastic_process as sp


class BrownianMotion(sp.StochasticProcess):
    def __init__(self, drift=0, diffusion=1, dimension=1):
        self.initial_value = 0
        super().__init__(drift, diffusion, self.initial_value, dimension)

    def _discretize(self, prev_value, dt, random_normal):
        return (
            prev_value + self.drift * dt + np.sqrt(self.diffusion * dt) * random_normal
        )

    def _validate_parameters(self):
        return super()._validate_parameters()

    def simulate(self, n_paths, n_steps, dt, return_times=False, seed=None):
        """Simulate paths of a brownian motion with drift and diffusion coefficent"""

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
        return self.drift * t

    def variance(self, t):
        return self.diffusion * t

    def skewness(self):
        return 0

    def kurtosis(self):
        return 3
