import numpy as np
from . import derivative_instrument as d
from ..models import pricing_model as p


class Forward(d.DerivativeInstrument):
    def __init__(self, forward_price: np.floating, **kwargs):
        super().__init__(**kwargs)
        self.forward_price = forward_price
        self._validate_parameters()

    def price(self, model: p.PricingModel) -> np.floating:
        return model.calculate_price(self)

    def delta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_delta(self)

    def theta(self, model: p.PricingModel) -> np.floating:
        return model.calculate_theta(self)

    def rho(self, model: p.PricingModel) -> np.floating:
        return model.calculate_rho(self)

    def epsilon(self, model: p.PricingModel) -> np.floating:
        return model.calculate_epsilon(self)

    def payoff(self, underlying_values):
        return underlying_values - self.forward_price

    def _validate_parameters(self):
        if self.forward_price < 0:
            raise ValueError("forward_price has to be non-negative")
        if self.underlying <= 0:
            raise ValueError("underlying has to be positive")
        if self.maturity <= 0:
            raise ValueError("maturity has to be positive")
        if self.payoff_fn is not None:
            raise ValueError("payoff_fn has to be None")
