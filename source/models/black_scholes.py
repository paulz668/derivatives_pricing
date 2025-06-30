import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from . import pricing_model as p
from ..instruments.derivative_instrument import DerivativeInstrument
from ..instruments.vanilla_options import EuropeanOption


class BlackScholes(p.PricingModel):
    def __init__(
        self,
        risk_free_rate: np.floating,
        volatility: np.floating,
        divident_yield: np.floating,
    ):
        """
        Parameters:
        -----------
        risk_free_rate: risk free rate (continuously compounded)
        divident_yield: annual divident yield (continuously compounded)
        volatility: volatility of underlying
        """
        self.r = risk_free_rate
        self.q = divident_yield
        self.sigma = volatility
        self._validate_parameters()

    def supports_instrument(self, instrument: DerivativeInstrument) -> np.bool_:
        if isinstance(instrument, EuropeanOption):
            return True
        else:
            return False

    def calculate_price(self, instrument: DerivativeInstrument):
        """
        Calculate the price of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - European options
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return S_0 * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(
                    -self.r * T
                ) * norm.cdf(d2)
            else:
                return K * np.exp(-self.r * T) * norm.cdf(-d2) - S_0 * np.exp(
                    -self.q * T
                ) * norm.cdf(-d1)

    def calculate_delta(self, instrument: DerivativeInstrument):
        """
        Calculate the delta (dV/dS) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - European options
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return np.exp(-self.q * T) * norm.cdf(d1)
            else:
                return -np.exp(-self.q * T) * norm.cdf(-d1)

    def calculate_vega(self, instrument: DerivativeInstrument):
        """
        Calculate the vega (dV/dσ) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - European options
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            return np.exp(-self.q * T) * norm.cdf(d1)

    def calculate_theta(self, instrument: DerivativeInstrument):
        """
        Calculate the theta (dV/dT) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - European options
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    -np.exp(-self.q * T)
                    * (S_0 * norm.pdf(d1) * self.sigma)
                    / (2 * np.sqrt(T))
                    - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
                    + self.q * S_0 * np.exp(-self.q * T) * norm.cdf(d1)
                )
            else:
                return (
                    -np.exp(-self.q * T)
                    * (S_0 * norm.pdf(d1) * self.sigma)
                    / (2 * np.sqrt(T))
                    + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
                    - self.q * S_0 * np.exp(-self.q * T) * norm.cdf(-d1)
                )

    def calculate_rho(self, instrument: DerivativeInstrument):
        """
        Calculate the theta (dV/dr) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - European options
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return K * T * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                return -K * T * np.exp(-self.r * T) * norm.cdf(-d2)

    def calculate_epsilon(self, instrument: DerivativeInstrument):
        """
        Calculate the theta (dV/dq) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - European options
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, EuropeanOption):
            S_0 = instrument.underlying
            K = instrument.strike
            T = instrument.maturity

            d1 = (np.log(S_0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return -S_0 * T * np.exp(-self.q * T) * norm.cdf(d1)
            else:
                return S_0 * T * np.exp(-self.q * T) * norm.cdf(-d1)

    def _validate_parameters(self):
        if self.r < 0:
            raise ValueError("risk_free_rate has to be non-negative")
        if self.sigma <= 0:
            raise ValueError("volatility has to be positive")
