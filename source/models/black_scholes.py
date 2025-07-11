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

        pricing_methods = {
            Forward: self._forward_price,
            EuropeanOption: self._european_option_price,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_price,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_price,
        }

        pricing_method = pricing_methods.get(type(instrument))
        if pricing_method:
            return pricing_method(instrument)
        raise NotImplementedError(
            f"Pricing not implemented for {type(instrument).__name__}"
        )

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

        delta_methods = {
            Forward: self._forward_delta,
            EuropeanOption: self._european_option_delta,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_delta,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_delta,
        }

        delta_method = delta_methods.get(type(instrument))
        if delta_method:
            return delta_method(instrument)
        raise NotImplementedError(
            f"Delta not implemented for {type(instrument).__name__}"
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

        vega_methods = {
            Forward: self._forward_vega,
            EuropeanOption: self._european_option_vega,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_vega,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_vega,
        }

        vega_method = vega_methods.get(type(instrument))
        if vega_method:
            return vega_method(instrument)
        raise NotImplementedError(
            f"Vega not implemented for {type(instrument).__name__}"
        )

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

        theta_methods = {
            Forward: self._forward_theta,
            EuropeanOption: self._european_option_theta,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_theta,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_theta,
        }

        theta_method = theta_methods.get(type(instrument))
        if theta_method:
            return theta_method(instrument)
        raise NotImplementedError(
            f"Theta not implemented for {type(instrument).__name__}"
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

        rho_methods = {
            Forward: self._forward_rho,
            EuropeanOption: self._european_option_rho,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_rho,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_rho,
        }

        rho_method = rho_methods.get(type(instrument))
        if rho_method:
            return rho_method(instrument)
        raise NotImplementedError(
            f"Rho not implemented for {type(instrument).__name__}"
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

        epsilon_methods = {
            Forward: self._forward_epsilon,
            EuropeanOption: self._european_option_epsilon,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_epsilon,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_epsilon,
        }

        epsilon_method = epsilon_methods.get(type(instrument))
        if epsilon_method:
            return epsilon_method(instrument)
        raise NotImplementedError(
            f"Epsilon not implemented for {type(instrument).__name__}"
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

        gamma_methods = {
            Forward: self._forward_gamma,
            EuropeanOption: self._european_option_gamma,
            CashOrNothingBinaryOption: self._cash_or_nothing_binary_option_gamma,
            AssetOrNothingBinaryOption: self._asset_or_nothing_binary_option_gamma,
        }

        gamma_method = gamma_methods.get(type(instrument))
        if gamma_method:
            return gamma_method(instrument)
        raise NotImplementedError(
            f"Gamma not implemented for {type(instrument).__name__}"
        )

    def _validate_parameters(self):
        if self.r < 0:
            raise ValueError("risk_free_rate has to be non-negative")
        if self.q < 0:
            raise ValueError("divident_yield has to be non-negative")
        if self.sigma <= 0:
            raise ValueError("volatility has to be positive")

    # Forward calculations
    def _forward_price(self, forward: Forward) -> np.floating:
        S = forward.underlying
        F = forward.forward_price
        T = forward.time_to_maturity

        return S * np.exp(-self.q * T) - np.exp(-self.r * T) * F

    def _forward_delta(self, forward: Forward) -> np.floating:
        T = forward.time_to_maturity

        return np.exp(-self.q * T)

    def _forward_vega(self, forward: Forward) -> np.floating:
        return 0

    def _forward_theta(self, forward: Forward) -> np.floating:
        S = forward.underlying
        F = forward.forward_price
        T = forward.time_to_maturity

        return self.q * S * np.exp(-self.q * T) - self.r * F * np.exp(-self.r * T)

    def _forward_rho(self, forward: Forward) -> np.floating:
        F = forward.forward_price
        T = forward.time_to_maturity

        return -T * np.exp(-self.r * T) * F

    def _forward_epsilon(self, forward: Forward) -> np.floating:
        S = forward.underlying
        T = forward.time_to_maturity

        return -T * S * np.exp(-self.q * T)

    def _forward_gamma(self, forward: Forward) -> np.floating:
        return 0

    # Option helpers
    def _d1(
        self,
        S: np.floating,
        K: np.floating,
        T: np.floating,
        r: Optional[np.floating] = None,
        q: Optional[np.floating] = None,
        sigma: Optional[np.floating] = None,
    ) -> np.floating:
        if r is None:
            r = self.r
        if q is None:
            q = self.q
        if sigma is None:
            sigma = self.sigma

        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def _d2(
        self,
        S: np.floating,
        K: np.floating,
        T: np.floating,
        r: Optional[np.floating] = None,
        q: Optional[np.floating] = None,
        sigma: Optional[np.floating] = None,
        d1: Optional[np.floating] = None,
    ) -> np.floating:
        if r is None:
            r = self.r
        if q is None:
            q = self.q
        if sigma is None:
            sigma = self.sigma

        if d1 is not None:
            return d1 - sigma * np.sqrt(T)
        else:
            return (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # European option calculations
    def _european_option_price(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(
                -self.r * T
            ) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * np.exp(
                -self.q * T
            ) * norm.cdf(-d1)

    def _european_option_delta(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
            return np.exp(-self.q * T) * norm.cdf(d1)
        else:
            return -np.exp(-self.q * T) * norm.cdf(-d1)

    def _european_option_vega(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        return S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T)

    def _european_option_theta(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return S * np.exp(-self.q * T) * (
                self.q * norm.cdf(d1) - (norm.pdf(d1) * self.sigma) / (2 * T)
            ) - self.r * np.exp(-self.r * T) * K * norm.cdf(d2)
        else:
            return S * np.exp(-self.q * T) * (
                -self.q * norm.cdf(-d1) - (norm.pdf(d1) * self.sigma) / (2 * T)
            ) + self.r * np.exp(-self.r * T) * K * norm.cdf(-d2)

    def _european_option_rho(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d2 = self._d2(S, K, T)

        if option.is_call:
            return K * T * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-self.r * T) * norm.cdf(-d2)

    def _european_option_epsilon(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
            return -S * T * np.exp(-self.q * T) * norm.cdf(d1)
        else:
            return S * T * np.exp(-self.q * T) * norm.cdf(-d1)

    def _european_option_gamma(self, option: EuropeanOption) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        return np.exp(-self.q * T) * norm.pdf(d1) / (S * self.sigma * np.sqrt(T))

    # Cash or Nothing Binary option calculations
    def _cash_or_nothing_binary_option_price(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d2 = self._d2(S, K, T)

        if option.is_call:
            return option.payout * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return option.payout * np.exp(-self.r * T) * norm.cdf(-d2)

    def _cash_or_nothing_binary_option_delta(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d2 = self._d2(S, K, T)

        if option.is_call:
            return (
                option.payout
                * np.exp(-self.r * T)
                * norm.pdf(d2)
                / (S * self.sigma * np.sqrt(T))
            )
        else:
            return (
                -option.payout
                * np.exp(-self.r * T)
                * norm.pdf(-d2)
                / (S * self.sigma * np.sqrt(T))
            )

    def _cash_or_nothing_binary_option_vega(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return -option.payout * np.exp(-self.r * T) * norm.pdf(d2) * d1 / self.sigma
        else:
            return option.payout * np.exp(-self.r * T) * norm.pdf(-d2) * d1 / self.sigma

    def _cash_or_nothing_binary_option_theta(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return (
                option.payout
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
                option.payout
                * np.exp(-self.r * T)
                * (
                    self.r * norm.cdf(-d2)
                    - norm.pdf(-d2)
                    * (np.log(S / K) - (self.r - self.q - 0.5 * self.sigma**2) * T)
                    / (2 * self.sigma * T**1.5)
                )
            )

    def _cash_or_nothing_binary_option_rho(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d2 = self._d2(S, K, T)

        if option.is_call:
            return (
                np.exp(-self.r * T)
                * option.payout
                * (-T * norm.cdf(d2) + norm.pdf(d2) * np.sqrt(T) / self.sigma)
            )
        else:
            return (
                np.exp(-self.r * T)
                * option.payout
                * (-T * norm.cdf(-d2) - norm.pdf(-d2) * np.sqrt(T) / self.sigma)
            )

    def _cash_or_nothing_binary_option_epsilon(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d2 = self._d2(S, K, T)

        if option.is_call:
            return (
                -np.exp(-self.r * T)
                * option.payout
                * norm.pdf(d2)
                * np.sqrt(T)
                / self.sigma
            )
        else:
            return (
                np.exp(-self.r * T)
                * option.payout
                * norm.pdf(-d2)
                * np.sqrt(T)
                / self.sigma
            )

    def _cash_or_nothing_binary_option_gamma(
        self, option: CashOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return (
                -np.exp(-self.r * T)
                * option.payout
                * norm.pdf(d2)
                * d1
                / (S**2 * self.sigma**2 * T)
            )
        else:
            return (
                np.exp(-self.r * T)
                * option.payout
                * norm.pdf(-d2)
                * d1
                / (S**2 * self.sigma**2 * T)
            )

    # Asset or Nothing Binary option calculations
    def _asset_or_nothing_binary_option_price(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
            return S * np.exp(-self.q * T) * norm.cdf(d1)
        else:
            return S * np.exp(-self.q * T) * norm.cdf(-d1)

    def _asset_or_nothing_binary_option_delta(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
            return np.exp(-self.q * T) * (
                norm.cdf(d1) + norm.pdf(d1) / (self.sigma * np.sqrt(T))
            )
        else:
            return np.exp(-self.q * T) * (
                norm.cdf(-d1) - norm.pdf(-d1) / (self.sigma * np.sqrt(T))
            )

    def _asset_or_nothing_binary_option_vega(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return -S * np.exp(-self.q * T) * norm.pdf(d1) * d2 / self.sigma
        else:
            return S * np.exp(-self.q * T) * norm.pdf(-d1) * d2 / self.sigma

    def _asset_or_nothing_binary_option_theta(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
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

    def _asset_or_nothing_binary_option_rho(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
            return S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T) / self.sigma
        else:
            return -S * np.exp(-self.q * T) * norm.pdf(-d1) * np.sqrt(T) / self.sigma

    def _asset_or_nothing_binary_option_epsilon(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)

        if option.is_call:
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

    def _asset_or_nothing_binary_option_gamma(
        self, option: AssetOrNothingBinaryOption
    ) -> np.floating:
        S = option.underlying
        K = option.strike
        T = option.time_to_maturity

        d1 = self._d1(S, K, T)
        d2 = self._d2(S, K, T, d1=d1)

        if option.is_call:
            return -np.exp(-self.q * T) * norm.pdf(d1) * d2 / (S * self.sigma**2 * T)
        else:
            return np.exp(-self.q * T) * norm.pdf(-d1) * d2 / (S * self.sigma**2 * T)
