import numpy as np
import stochastic_process as sp

from _collections_abc import Callable


class ItoProcess(sp.StochasticProcess):
    def __init__(self, drift=0, diffusion=1, initial_value=1, dimension=1):
        super().__init__(drift, diffusion, initial_value, dimension)

    def _discretize(self, prev_value, t, dt, random_normal, scheme):
        if scheme == "first order Euler":
            return (
                prev_value
                + self.drift(t, prev_value) * dt
                + self.diffusion(t, prev_value) * np.sqrt(dt) * random_normal
            )
        elif scheme == "Millstein":
            raise NotImplementedError
        elif scheme == "secodn order Euler":
            raise NotImplementedError
        else:
            raise ValueError(
                "scheme must be 'first order Euler', 'Millstein' or 'second order Euler'"
            )

    def _check_lipschitz(self):
        raise NotImplementedError

    def _check_linear_growth(self):
        raise NotImplementedError

    def _validate_parameters(self):
        if not isinstance(
            self.drift, Callable[[np.floating, np.floating], np.floating]
        ):
            raise ValueError(
                "Drift term has to be a function taking two floats and returning one float"
            )
        if not isinstance(
            self.diffusion, Callable[[np.floating, np.floating], np.floating]
        ):
            raise ValueError(
                "Diffusion term has to be a function taking two floats and returning one float"
            )
        if self.dimension < 1:
            raise ValueError("Dimension must be ≥ 1")

    def simulate(self, n_paths, n_steps, dt, scheme, return_times=False, seed=None):
        """Simulate paths of an Ito process"""

        rng = np.random.default_rng(seed=seed)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.initial_value

        if scheme == "first order Euler" or scheme == "Millstein":
            normal_sample = rng.standard_normal((n_paths, n_steps))
        elif scheme == "second order Euler":
            pass
            # normal_sample = rng.multivariate_normal()
        else:
            raise ValueError(
                "scheme must be 'first order Euler', 'Millstein' or 'second order Euler'"
            )

        for t in range(1, n_steps + 1):
            paths[:, t] = self._discretize(
                paths[:, t - 1], t * dt, dt, normal_sample[:, t - 1], scheme=scheme
            )

        if return_times:
            times = np.linspace(0, n_steps * dt, n_steps + 1)
            return paths, times

        return paths

    def mean(self, t):
        raise NotImplementedError

    def variance(self, t):
        raise NotImplementedError

    def skewness(self, t):
        raise NotImplementedError

    def kurtosis(self, t):
        raise NotImplementedError
