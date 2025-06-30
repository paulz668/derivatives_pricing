from abc import ABC, abstractmethod
from typing import Optional, Union, Callable
import numpy as np
import numpy.typing as npt


class DerivativeInstrument(ABC):
    def __init__(
        self,
        underlying: Union[np.floating, npt.ArrayLike],
        maturity: np.floating,
        payoff_fn: Optional[Callable] = None,
    ):
        """
        Parameters:
        -----------
        underlying : Current value(s) of underlying asset(s)
        maturity : Time to maturity in years
        payoff_fn : Function defining the instrument's payoff
        """
        self.underlying = underlying
        self.maturity = maturity
        self.payoff_fn = payoff_fn

    @abstractmethod
    def price(self, model) -> np.floating:
        """Price the instrument using the given model"""
        pass

    def payoff(self, underlying_values: npt.ArrayLike) -> npt.ArrayLike:
        """Calculate payoff for given underlying values"""
        if self.payoff_fn is not None:
            return self.payoff_fn(underlying_values)
        raise NotImplementedError("Payoff function not defined")

    @abstractmethod
    def _validate_parameters(self):
        """Validate the given parameters"""
        pass
