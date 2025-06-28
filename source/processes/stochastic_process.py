import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Optional, Union
from _collections_abc import Callable


class StochasticProcess(ABC):
    def __init__(
        self,
        drift: Union[
            np.floating, Callable[[np.floating, np.floating], np.floating]
        ] = 0.0,
        diffusion: Union[
            np.floating, Callable[[np.floating, np.floating], np.floating]
        ] = 1.0,
        initial_value: np.floating = 1.0,
        dimension: np.integer = 1,
    ):
        """
        Parameters:
        -----------
        drift : float
            Mean/drift coefficient
        diffusion : float
            Volatility/diffusion coefficient
        initial_value : float
            Starting value of the process
        dimension : int
            Dimensionality of the process (for multi-dimensional processes)
        """
        self.drift = drift
        self.diffusion = diffusion
        self.initial_value = initial_value
        self.dimension = dimension
        self._validate_parameters()

    @abstractmethod
    def simulate(
        self,
        n_paths: np.integer,
        n_steps: np.integer,
        dt: np.floating,
        return_times: np.bool_ = False,
        seed: Optional[np.integer] = None,
    ):
        """
        Simulate paths of the stochastic process

        Parameters:
        -----------
        n_paths : int
            Number of paths to simulate
        n_steps : int
            Number of time steps
        dt : float
            Time step size
        return_times : bool
            If True, returns time points along with paths
        seed : Optional[int]
            Random seed for reproducibility

        Returns:
        --------
        paths : ndarray
            Simulated paths (shape: n_paths x (n_steps + 1))
        times : ndarray (optional)
            Time points if return_times=True
        """
        pass

    @abstractmethod
    def _discretize(
        self,
        prev_value: np.floating,
        dt: np.floating,
        random_normal: Union[np.floating, npt.ArrayLike],
    ):
        """
        Core discretization scheme for the process

        Parameters:
        -----------
        prev_value : float
            Value at previous time step
        dt : float
            Time step size
        random_normal : float
            Random normal variable for the step

        Returns:
        --------
        new_value : float
            Value at next time step
        """
        pass

    def _validate_parameters(self):
        """Validate process parameters"""
        if self.diffusion <= 0:
            raise ValueError("Diffusion coefficient must be positive")
        if self.dimension < 1:
            raise ValueError("Dimension must be ≥ 1")

    @abstractmethod
    def mean(self, t: np.floating) -> np.floating:
        """Theoretical mean at time t"""
        pass

    @abstractmethod
    def variance(self, t: np.floating) -> np.floating:
        """Theoretical variance at time t"""
        pass

    @abstractmethod
    def skewness(self, t: np.floating) -> np.floating:
        """Theoretical skewness at time t"""
        pass

    @abstractmethod
    def kurtosis(self, t: np.floating) -> np.floating:
        """Theoretical kurtosis at time t"""
        pass
