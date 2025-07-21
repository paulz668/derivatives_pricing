import numpy as np

from . import pricing_model as p

from ..instruments.derivative_instrument import DerivativeInstrument
from ..instruments.forward import Forward
from ..instruments.european_option import EuropeanOption
from ..instruments.binary_option import *
from ..instruments.asian_option import *

from ..utils.pde_solver import theta_method_pde_solver


class Heston(p.PricingModel):
    def __init__(
        self,
        risk_free_rate: np.floating,
        mean_reversion_rate: np.floating,
        mean_volatility: np.floating,
        vol_of_vol: np.floating,
        correlation: np.floating,
    ):
        """
        Parameters:
        -----------
        risk_free_rate: risk_free_rate (continuously compounded)
        mean_reversion_rate: mean reversion rate of the volatility process
        mean_volatility: mean volality of the volatility process
        vol_of _vol: volatility term of the volatility process
        corrlation: correlation of uderlying brownian motions
        """
        self.r = risk_free_rate
        self.kappa = mean_reversion_rate
        self.v_bar = mean_volatility
        self.nu = vol_of_vol
        self.rho = correlation
        self._validate_parameters()

    def supports_instrument(self, instrument: DerivativeInstrument) -> np.bool_:
        """Check if instrument is supported"""
        supported_instruments = {
            "Forward": False,
            "EuropeanOption": False,
            "CashOrNothingBinaryOption": False,
            "AssetOrNothingBinaryOption": False,
            "TestInstrument": False,
        }

        return supported_instruments[instrument.__class__.__name__]

    def calculate_price(self, instrument: DerivativeInstrument) -> np.floating:
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
            Forward: None,
            EuropeanOption: None,
            CashOrNothingBinaryOption: None,
            AssetOrNothingBinaryOption: None,
            ArithmeticAverageFixedStrikeAsianOption: None,
            ArithmeticAverageFloatingStrikeAsianOption: None,
            GeometricAverageFixedStrikeAsianOption: None,
            GeometricAverageFloatingStrikeAsianOption: None,
        }

        pricing_method = pricing_methods.get(type(instrument))
        if pricing_method:
            return pricing_method(instrument)
        raise NotImplementedError(
            f"Pricing not implemented for {type(instrument).__name__}"
        )

    def calculate_delta(self, instrument: DerivativeInstrument) -> np.floating:
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
            Forward: None,
            EuropeanOption: None,
            CashOrNothingBinaryOption: None,
            AssetOrNothingBinaryOption: None,
            ArithmeticAverageFixedStrikeAsianOption: None,
            ArithmeticAverageFloatingStrikeAsianOption: None,
            GeometricAverageFixedStrikeAsianOption: None,
            GeometricAverageFloatingStrikeAsianOption: None,
        }

        delta_method = delta_methods.get(type(instrument))
        if delta_method:
            return delta_method(instrument)
        raise NotImplementedError(
            f"Delta not implemented for {type(instrument).__name__}"
        )

    def calculate_vega(self, instrument: DerivativeInstrument) -> np.floating:
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
            Forward: None,
            EuropeanOption: None,
            CashOrNothingBinaryOption: None,
            AssetOrNothingBinaryOption: None,
            ArithmeticAverageFixedStrikeAsianOption: None,
            ArithmeticAverageFloatingStrikeAsianOption: None,
            GeometricAverageFixedStrikeAsianOption: None,
            GeometricAverageFloatingStrikeAsianOption: None,
        }

        vega_method = vega_methods.get(type(instrument))
        if vega_method:
            return vega_method(instrument)
        raise NotImplementedError(
            f"Vega not implemented for {type(instrument).__name__}"
        )

    def calculate_theta(self, instrument: DerivativeInstrument) -> np.floating:
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
            Forward: None,
            EuropeanOption: None,
            CashOrNothingBinaryOption: None,
            AssetOrNothingBinaryOption: None,
            ArithmeticAverageFixedStrikeAsianOption: None,
            ArithmeticAverageFloatingStrikeAsianOption: None,
            GeometricAverageFixedStrikeAsianOption: None,
            GeometricAverageFloatingStrikeAsianOption: None,
        }

        theta_method = theta_methods.get(type(instrument))
        if theta_method:
            return theta_method(instrument)
        raise NotImplementedError(
            f"Theta not implemented for {type(instrument).__name__}"
        )

    def calculate_rho(self, instrument: DerivativeInstrument) -> np.floating:
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
            Forward: None,
            EuropeanOption: None,
            CashOrNothingBinaryOption: None,
            AssetOrNothingBinaryOption: None,
            ArithmeticAverageFixedStrikeAsianOption: None,
            ArithmeticAverageFloatingStrikeAsianOption: None,
            GeometricAverageFixedStrikeAsianOption: None,
            GeometricAverageFloatingStrikeAsianOption: None,
        }

        rho_method = rho_methods.get(type(instrument))
        if rho_method:
            return rho_method(instrument)
        raise NotImplementedError(
            f"Rho not implemented for {type(instrument).__name__}"
        )

    def calculate_gamma(self, instrument: DerivativeInstrument) -> np.floating:
        """
        Calculate the gamma (d^2V/dS^2) of a derivative instrument in the Black_Scholes world
        Currently supported instruments:
        - Forward
        - European option
        - Cash or nothing binary option
        """
        self.validate_inputs(instrument=instrument)

        gamma_methods = {
            Forward: None,
            EuropeanOption: None,
            CashOrNothingBinaryOption: None,
            AssetOrNothingBinaryOption: None,
            ArithmeticAverageFixedStrikeAsianOption: None,
            ArithmeticAverageFloatingStrikeAsianOption: None,
            GeometricAverageFixedStrikeAsianOption: None,
            GeometricAverageFloatingStrikeAsianOption: None,
        }

        gamma_method = gamma_methods.get(type(instrument))
        if gamma_method:
            return gamma_method(instrument)
        raise NotImplementedError(
            f"Gamma not implemented for {type(instrument).__name__}"
        )

    def _validate_parameters(self):
        if self.r < 0:
            raise ValueError("risk_free_rate has to be positive")
        if self.kappa < 0:
            raise ValueError("mean_reversion_rate has to be positive")
        if self.v_bar < 0:
            raise ValueError("mean_volatility has to be positive")
        if self.nu < 0:
            raise ValueError("vol_of_vol has to be positive")
        if self.rho < -1 or self.rho > 1:
            raise ValueError("correlation has to be in [-1, 1]")
