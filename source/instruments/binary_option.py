import numpy as np
from typing import Optional

from . import derivative_instrument as d
from ..models import pricing_model as p


class CashOrNothingBinaryOption(d.DerivativeInstrument):
    def __init__(
        self,
        underlying: np.floating,
        strike: np.floating,
        time_to_maturity: np.floating,
        payout: Optional[np.floating],
        is_call: np.bool_,
    ):
        super().__init__(underlying, time_to_maturity)
        self.strike = strike
        self.is_call = is_call
        self.payout = 1 if payout is None else payout
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
            return np.where(underlying_values >= self.strike, self.payout, 0)
        else:
            return np.where(underlying_values < self.strike, self.payout, 0)

    def _validate_parameters(self):
        if self.strike < 0:
            raise ValueError("strike has to be non-negative")
        if isinstance(self.is_call, np.bool_):
            raise ValueError("is_call has to be a boolean")
        if self.payout < 0:
            raise ValueError("payout has to be positive")
        if self.underlying <= 0:
            raise ValueError("underlying has to be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")


class AssetOrNothingBinaryOption(d.DerivativeInstrument):
    def __init__(
        self,
        underlying: np.floating,
        strike: np.floating,
        time_to_maturity: np.floating,
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
            return np.where(underlying_values >= self.strike, underlying_values, 0)
        else:
            return np.where(underlying_values < self.strike, underlying_values, 0)

    def _validate_parameters(self):
        if self.strike < 0:
            raise ValueError("strike has to be non-negative")
        if isinstance(self.is_call, np.bool_):
            raise ValueError("is_call has to be a boolean")
        if self.underlying <= 0:
            raise ValueError("underlying has to be positive")
        if self.time_to_maturity <= 0:
            raise ValueError("maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")
