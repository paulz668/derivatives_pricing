import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from . import pricing_model as p
from ..instruments.european_option import EuropeanOption


class BlackScholes(p.PricingModel):
    def __init__(self, risk_free_rate: np.floating, volatility: np.floating):
        """
        Parameters:
        -----------
        risk_free_rate: rate of return of the underlying (under the risk neutral probability measure)
        volatility: volatility of underlying
        """
        self.r = risk_free_rate
        self.sigma = volatility
        self._validate_parameters()

    def supports_instrument(self, instrument) -> np.bool_:
        if isinstance(instrument, EuropeanOption):
            return True
        else:
            return False
        
    def calculate_price(self, instrument):
        self.validate_inputs()
        
        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return S_0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-self.r * T) * norm.cdf(d2) - S_0 * norm.cdf(d1) 

    def _validate_parameters(self):
        if self.r < 0:
            raise ValueError("risk_free_rate has to be non-negative")
        if self.sigma <= 0:
            raise ValueError("volatility has to be positive")
