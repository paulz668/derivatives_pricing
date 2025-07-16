import numpy as np

from numpy.typing import ArrayLike
from typing import Optional
from . import derivative_instrument as d
from ..models import pricing_model as p


class ArithmeticAverageFixedStrikeAsianOption(d.DerivativeInstrument):
    def __init__(
        self,
        underlying: ArrayLike,
        strike: np.floating,
        t: np.floating,
        T: np.floating,
        is_call: np.bool_,
    ):
        super().__init__(underlying, T - t)
        self.strike = strike
        self.is_call = is_call
        self.current_time = t
        self.term = T
        self._validate_parameters()

    def price(self, model: p.PricingModel) -> np.floating:
        return model.calculate_price(self)

    def delta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_delta(self)

    def vega(self, model: p.PricingModel) -> np.floating:
        return model.calculate_vega(self)

    def theta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_theta(self)

    def rho(self, model: p.PricingModel) -> np.floating:
        return model.calculate_rho(self)

    def epsilon(self, model: p.PricingModel) -> np.floating:
        return model.calculate_epsilon(self)

    def gamma(self, model: p.PricingModel) -> np.floating:
        return model.calculate_gamma(self)

    def payoff(self, underlying_values):
        if self.is_call:
            return np.maximum(np.mean(underlying_values, 0) - self.strike, 0)
        return np.maximum(self.strike - np.mean(underlying_values, 0), 0)

    def _validate_parameters(self):
        if self.strike < 0:
            raise ValueError("strike has to be non-negative")
        if isinstance(self.is_call, np.bool_):
            raise ValueError("is_call has to be a boolean")
        if np.all(self.underlying > 0):
            raise ValueError("underlying has to be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")


class ArithmeticAverageFloatingStrikeAsianOption(d.DerivativeInstrument):
    def __init__(
        self,
        underlying: ArrayLike,
        strike: Optional[ArrayLike],
        t: np.floating,
        T: np.floating,
        is_call: np.bool_,
    ):
        super().__init__(underlying, T - t)
        self.strike = strike
        self.is_call = is_call
        self.current_time = t
        self.term = T
        self._validate_parameters()

    def price(self, model: p.PricingModel) -> np.floating:
        return model.calculate_price(self)

    def delta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_delta(self)

    def vega(self, model: p.PricingModel) -> np.floating:
        return model.calculate_vega(self)

    def theta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_theta(self)

    def rho(self, model: p.PricingModel) -> np.floating:
        return model.calculate_rho(self)

    def epsilon(self, model: p.PricingModel) -> np.floating:
        return model.calculate_epsilon(self)

    def gamma(self, model: p.PricingModel) -> np.floating:
        return model.calculate_gamma(self)

    def payoff(self, underlying_values):
        if self.strike is not None:
            strike = np.mean(np.concatenate(self.strike, underlying_values))
        else:
            strike = np.mean(underlying_values)
        if self.is_call:
            return np.maximum(underlying_values[-1] - strike, 0)
        return np.maximum(strike - underlying_values[-1], 0)

    def _validate_parameters(self):
        if np.all(self.strike > 0):
            raise ValueError("strike has to be positive")
        if isinstance(self.is_call, np.bool_):
            raise ValueError("is_call has to be a boolean")
        if np.all(self.underlying > 0):
            raise ValueError("underlying has to be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")


class GeometricAverageFixedStrikeAsianOption(d.DerivativeInstrument):
    def __init__(
        self,
        underlying: ArrayLike,
        strike: np.floating,
        time_to_maturity,
        is_call: np.bool_,
    ):
        super().__init__(underlying, time_to_maturity)
        self.strike = strike
        self.is_call = is_call
        self._validate_parameters()

    def price(self, model: p.PricingModel) -> np.floating:
        return model.calculate_price(self)

    def delta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_delta(self)

    def vega(self, model: p.PricingModel) -> np.floating:
        return model.calculate_vega(self)

    def theta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_theta(self)

    def rho(self, model: p.PricingModel) -> np.floating:
        return model.calculate_rho(self)

    def epsilon(self, model: p.PricingModel) -> np.floating:
        return model.calculate_epsilon(self)

    def gamma(self, model: p.PricingModel) -> np.floating:
        return model.calculate_gamma(self)

    def payoff(self, underlying_values):
        if self.is_call:
            return np.maximum(
                np.exp(np.mean(np.log(underlying_values), 0)) - self.strike, 0
            )
        return np.maximum(
            self.strike - np.exp(np.mean(np.log(underlying_values), 0)), 0
        )

    def _validate_parameters(self):
        if self.strike < 0:
            raise ValueError("strike has to be non-negative")
        if isinstance(self.is_call, np.bool_):
            raise ValueError("is_call has to be a boolean")
        if np.all(self.underlying > 0):
            raise ValueError("underlying has to be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")


class GeometricAverageFloatingStrikeAsianOption(d.DerivativeInstrument):
    def __init__(
        self,
        underlying: ArrayLike,
        strike: Optional[ArrayLike],
        time_to_maturity,
        is_call: np.bool_,
    ):
        super().__init__(underlying, time_to_maturity)
        self.strike = strike
        self.is_call = is_call
        self._validate_parameters()

    def price(self, model: p.PricingModel) -> np.floating:
        return model.calculate_price(self)

    def delta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_delta(self)

    def vega(self, model: p.PricingModel) -> np.floating:
        return model.calculate_vega(self)

    def theta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_theta(self)

    def rho(self, model: p.PricingModel) -> np.floating:
        return model.calculate_rho(self)

    def epsilon(self, model: p.PricingModel) -> np.floating:
        return model.calculate_epsilon(self)

    def gamma(self, model: p.PricingModel) -> np.floating:
        return model.calculate_gamma(self)

    def payoff(self, underlying_values):
        if self.strike is not None:
            strike = np.exp(
                np.mean(np.log(np.concatenate(self.strike, underlying_values)))
            )
        else:
            strike = np.exp(np.mean(np.log(underlying_values)))
        if self.is_call:
            return np.maximum(underlying_values[-1] - strike, 0)
        return np.maximum(strike - underlying_values[-1], 0)

    def _validate_parameters(self):
        if np.all(self.strike > 0):
            raise ValueError("strike has to be positive")
        if isinstance(self.is_call, np.bool_):
            raise ValueError("is_call has to be a boolean")
        if np.all(self.underlying > 0):
            raise ValueError("underlying has to be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("time_to_maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")
