import numpy as np

from . import pricing_model as p

from ..instruments.derivative_instrument import DerivativeInstrument
from ..instruments.forward import Forward
from ..instruments.european_option import EuropeanOption
from ..instruments.binary_option import *
from ..instruments.asian_option import *

from ..utils.pde_solver import theta_method_pde_solver


class CEV(p.PricingModel):
    def __init__(
        self, risk_free_rate: np.floating, scale: np.floating, elasticity: np.floating
    ):
        """
        Parameters:
        -----------
        drift: fixes drift of underlying process
        scale: fixes at-the-money volatility level
        elasticity: elasticity parameter of local volatility
        """
        self.r = risk_free_rate
        self.delta = scale
        self.beta = elasticity
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
            raise ValueError("risk_free_rate has to be non-negative")
        if self.delta <= 0:
            raise ValueError("scale has to be positive")
        if self.beta <= -1 or self.beta >= 1:
            raise ValueError("elasticity has to be in (-1, 1)")
