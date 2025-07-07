import numpy as np
from scipy.stats import norm

from . import pricing_model as p
from ..instruments.derivative_instrument import DerivativeInstrument
from ..instruments.forward import Forward
from ..instruments.european_option import EuropeanOption
from ..instruments.binary_option import *


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
        """Check if instrument is supported"""
        supported_instruments = {
            "Forward": True,
            "EuropeanOption": True,
            "CashOrNothingBinaryOption": True,
            "AssetOrNothingBinaryOption": True,
            "TestInstrument": False,
        }

        return supported_instruments[instrument.__class__.__name__]

    def calculate_price(self, instrument: DerivativeInstrument):
        """
        Calculate the price of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        - Asset or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            S = instrument.underlying
            F = instrument.forward_price
            T = instrument.time_to_maturity

            return S * np.exp(-self.q * T) - np.exp(-self.r * T) * F

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(
                    -self.r * T
                ) * norm.cdf(d2)
            else:
                return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * np.exp(
                    -self.q * T
                ) * norm.cdf(-d1)

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return instrument.payout * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                return instrument.payout * np.exp(-self.r * T) * norm.cdf(-d2)

        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return S * np.exp(-self.q * T) * norm.cdf(d1)
            else:
                return S * np.exp(-self.q * T) * norm.cdf(-d1)

    def calculate_delta(self, instrument: DerivativeInstrument):
        """
        Calculate the delta (dV/dS) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        - Asset or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            T = instrument.time_to_maturity

            return np.exp(-self.q * T)

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return np.exp(-self.q * T) * norm.cdf(d1)
            else:
                return -np.exp(-self.q * T) * norm.cdf(-d1)

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    instrument.payout
                    * np.exp(-self.r * T)
                    * norm.pdf(d2)
                    / (S * self.sigma * np.sqrt(T))
                )
            else:
                return (
                    -instrument.payout
                    * np.exp(-self.r * T)
                    * norm.pdf(-d2)
                    / (S * self.sigma * np.sqrt(T))
                )

        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return np.exp(-self.q * T) * (
                    norm.cdf(d1) + norm.pdf(d1) / (self.sigma * np.sqrt(T))
                )
            else:
                return np.exp(-self.q * T) * (
                    norm.cdf(-d1) - norm.pdf(-d1) / (self.sigma * np.sqrt(T))
                )

    def calculate_vega(self, instrument: DerivativeInstrument):
        """
        Calculate the vega (dV/dσ) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        - Asset or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            return 0

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            return S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    -instrument.payout
                    * np.exp(-self.r * T)
                    * norm.pdf(d2)
                    * d1
                    / self.sigma
                )
            else:
                return (
                    instrument.payout
                    * np.exp(-self.r * T)
                    * norm.pdf(-d2)
                    * d1
                    / self.sigma
                )

        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return -S * np.exp(-self.q * T) * norm.pdf(d1) * d2 / self.sigma
            else:
                return S * np.exp(-self.q * T) * norm.pdf(-d1) * d2 / self.sigma

    def calculate_theta(self, instrument: DerivativeInstrument):
        """
        Calculate the theta (dV/dT) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        - Asset or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            S = instrument.underlying
            F = instrument.forward_price
            T = instrument.time_to_maturity

            return self.q * S * np.exp(-self.q * T) - self.r * F * np.exp(-self.r * T)

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return S * np.exp(-self.q * T) * (
                    self.q * norm.cdf(d1) - (norm.pdf(d1) * self.sigma) / (2 * T)
                ) - self.r * np.exp(-self.r * T) * K * norm.cdf(d2)
            else:
                return S * np.exp(-self.q * T) * (
                    -self.q * norm.cdf(-d1) - (norm.pdf(d1) * self.sigma) / (2 * T)
                ) + self.r * np.exp(-self.r * T) * K * norm.cdf(-d2)

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    instrument.payout
                    * np.exp(-self.r * T)
                    * (
                        self.r * norm.cdf(d2)
                        + norm.pdf(d2)
                        * (np.log(S / K) - (self.r - self.q - 0.5 * self.sigma**2) * T)
                        / (2 * self.sigma * T**1.5)
                    )
                )
            else:
                return (
                    instrument.payout
                    * np.exp(-self.r * T)
                    * (
                        self.r * norm.cdf(-d2)
                        - norm.pdf(-d2)
                        * (np.log(S / K) - (self.r - self.q - 0.5 * self.sigma**2) * T)
                        / (2 * self.sigma * T**1.5)
                    )
                )

        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return (
                    S
                    * np.exp(-self.q * T)
                    * (
                        self.q * norm.cdf(d1)
                        + norm.pdf(d1)
                        * (np.log(S / K) - (self.r - self.q + 0.5 * self.sigma**2) * T)
                        / (2 * self.sigma * T**1.5)
                    )
                )
            else:
                return (
                    S
                    * np.exp(-self.q * T)
                    * (
                        self.q * norm.cdf(-d1)
                        - norm.pdf(-d1)
                        * (np.log(S / K) - (self.r - self.q + 0.5 * self.sigma**2) * T)
                        / (2 * self.sigma * T**1.5)
                    )
                )

    def calculate_rho(self, instrument: DerivativeInstrument):
        """
        Calculate the theta (dV/dr) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        - Asset or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            F = instrument.forward_price
            T = instrument.time_to_maturity

            return -T * np.exp(-self.r * T) * F

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return K * T * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                return -K * T * np.exp(-self.r * T) * norm.cdf(-d2)

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    np.exp(-self.r * T)
                    * instrument.payout
                    * (-T * norm.cdf(d2) + norm.pdf(d2) * np.sqrt(T) / self.sigma)
                )
            else:
                return (
                    np.exp(-self.r * T)
                    * instrument.payout
                    * (-T * norm.cdf(-d2) - norm.pdf(-d2) * np.sqrt(T) / self.sigma)
                )

        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T) / self.sigma
            else:
                return (
                    -S * np.exp(-self.q * T) * norm.pdf(-d1) * np.sqrt(T) / self.sigma
                )

    def calculate_epsilon(self, instrument: DerivativeInstrument):
        """
        Calculate the theta (dV/dq) of a derivative instrument in the Black-Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        - Asset or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            S = instrument.underlying
            T = instrument.time_to_maturity

            return -T * S * np.exp(-self.q * T)

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return -S * T * np.exp(-self.q * T) * norm.cdf(d1)
            else:
                return S * T * np.exp(-self.q * T) * norm.cdf(-d1)

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    -np.exp(-self.r * T)
                    * instrument.payout
                    * norm.pdf(d2)
                    * np.sqrt(T)
                    / self.sigma
                )
            else:
                return (
                    np.exp(-self.r * T)
                    * instrument.payout
                    * norm.pdf(-d2)
                    * np.sqrt(T)
                    / self.sigma
                )
        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return (
                    S
                    * np.exp(-self.q * T)
                    * (-norm.pdf(d1) * np.sqrt(T) / self.sigma - T * norm.cdf(d1))
                )
            else:
                return (
                    S
                    * np.exp(-self.q * T)
                    * (norm.pdf(-d1) * np.sqrt(T) / self.sigma - T * norm.cdf(-d1))
                )

    def calculate_gamma(self, instrument: DerivativeInstrument):
        """
        Calculate the gamma (d^2V/dS^2) of a derivative instrument in the Black_Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        if isinstance(instrument, Forward):
            return 0

        if isinstance(instrument, EuropeanOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            return np.exp(-self.q * T) * norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

        if isinstance(instrument, CashOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )
            d2 = d1 - self.sigma * np.sqrt(T)

            if instrument.is_call:
                return (
                    -np.exp(-self.r * T)
                    * instrument.payout
                    * norm.pdf(d2)
                    * d1
                    / (S**2 * self.sigma**2 * T)
                )
            else:
                return (
                    np.exp(-self.r * T)
                    * instrument.payout
                    * norm.pdf(-d2)
                    * d1
                    / (S**2 * self.sigma**2 * T)
                )

        if isinstance(instrument, AssetOrNothingBinaryOption):
            S = instrument.underlying
            K = instrument.strike
            T = instrument.time_to_maturity

            d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
                self.sigma * np.sqrt(T)
            )

            if instrument.is_call:
                return (
                    -np.exp(-self.q * T) * norm.pdf(d1) * d2 / (S * self.sigma**2 * T)
                )
            else:
                return (
                    np.exp(-self.q * T) * norm.pdf(-d1) * d2 / (S * self.sigma**2 * T)
                )

    def _validate_parameters(self):
        if self.r < 0:
            raise ValueError("risk_free_rate has to be non-negative")
        if self.q < 0:
            raise ValueError("divident_yield has to be non-negative")
        if self.sigma <= 0:
            raise ValueError("volatility has to be positive")
