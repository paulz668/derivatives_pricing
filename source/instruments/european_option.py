import numpy as np
from . import derivative_instrument as d
from ..models import pricing_model as p


class EuropeanOption(d.DerivativeInstrument):
    def __init__(self, strike: np.floating, is_call: np.bool_, **kwargs):
        super().__init__(**kwargs)
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
            return np.maximum(underlying_values - self.strike, 0)
        return np.maximum(self.strike - underlying_values, 0)

    def _validate_parameters(self):
        if self.strike < 0:
            raise ValueError("strike has to be non-negative")
        if self.underlying <= 0:
            raise ValueError("underlying has to be positive")
        if self.maturity <= 0:
            raise ValueError("maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")
