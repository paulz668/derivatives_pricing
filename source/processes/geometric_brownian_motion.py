import numpy as np
import stochastic_process as sp


class GeometricBrownianMotion(sp.StochasticProcess):
    def __init__(self, drift=0, diffusion=1, initial_value=1, dimension=1):
        super().__init__(drift, diffusion, initial_value, dimension)

    def _discretize(self, prev_value, dt, random_normal):
        return prev_value * np.exp(
            (self.drift - 0.5 * self.diffusion) * dt
            + np.sqrt(self.diffusion * dt) * random_normal
        )

    def _validate_parameters(self):
        return super()._validate_parameters()

    def simulate(self, n_paths, n_steps, dt, return_times=False, seed=None):
        """Simulate paths of a geometric brownian motion"""

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
        return self.initial_value * np.exp(self.drift * t)

    def variance(self, t):
        return (
            self.initial_value**2
            * np.exp(2 * self.drift * t)
            * (np.exp(self.diffusion * t) - 1)
        )

    def skewness(self, t):
        return (np.exp(self.diffusion * t) - 2) * np.sqrt(
            np.exp(self.diffusion * t) - 1
        )

    def kurtosis(self, t):
        return (
            np.exp(4 * self.diffusion * t)
            + 2 * np.exp(3 * self.diffusion * t)
            + 3 * np.exp(2 * self.diffusion * t)
            - 3
        )
