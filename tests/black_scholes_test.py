import numpy as np
import pytest

from source.models.black_scholes import BlackScholes
from source.instruments.forward import Forward
from source.instruments.european_option import EuropeanOption
from source.instruments.binary_option import *


class TestBlackScholes:
    # Initialization Tests
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2, divident_yield=0)
        assert bs.r == 0.02
        assert bs.sigma == 0.2
        assert bs.q == 0

    # support_instrument Method Test
    def test_forward_support(self):
        """Test supports_instrument method produces expected output with forward"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2, divident_yield=0)
        instrument = Forward(100, 100, 1)
        assert bs.supports_instrument(instrument) == True

    def test_european_option_support(self):
        """Test supports_instrument method produces expected output with european option"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2, divident_yield=0)
        instrument = EuropeanOption(100, 100, 1, True)
        assert bs.supports_instrument(instrument) == True

    def test_cash_or_nothing_binary_option_support(self):
        """Test supports_instrument method produces expected output with cash or nothing binary option"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2, divident_yield=0)
        instrument = CashOrNothingBinaryOption(100, 100, 1, 100, True)
        assert bs.supports_instrument(instrument) == True

    def test_asset_or_nothing_binary_option_support(self):
        """Test supports_instrument method produces expected output with asset or nothing binary option"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2, divident_yield=0)
        instrument = AssetOrNothingBinaryOption(100, 100, 1, True)
        assert bs.supports_instrument(instrument) == True

    # Validation Method Test
    # def test_validate_inputs(self):
    #     """Test validation_inputs method produces expected output"""
    #     bs = BlackScholes(risk_free_rate=0.02, volatility=0.2, divident_yield=0)
    #     instrument = EuropeanOption(100, 100, 1, True)
    #     assert bs.validate_inputs(instrument) == True

    # Parameter Validation Tests
    def test_negative_risk_free_rate_raises_error(self):
        """Test that negative risk free rate raises error"""
        with pytest.raises(ValueError):
            BlackScholes(risk_free_rate=-0.02, volatility=0.2, divident_yield=0)

    def test_zero_volatility_raises_error(self):
        """Test that zero volatility raises error"""
        with pytest.raises(ValueError):
            BlackScholes(risk_free_rate=0.02, volatility=0, divident_yield=0)

    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises error"""
        with pytest.raises(ValueError):
            BlackScholes(risk_free_rate=0.02, volatility=-0.2, divident_yield=0)

    def test_negative_divident_yield_raises_error(self):
        """Test that negative divident yield raises error"""
        with pytest.raises(ValueError):
            BlackScholes(risk_free_rate=0.02, volatility=-0.2, divident_yield=-0.1)

    # Price calculation test

    def test_calculate_forward_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = Forward(100, 104.6027859908717, 1.5)
        np.testing.assert_allclose(bs.calculate_price(instrument), 0)

    def test_calculate_european_call_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = EuropeanOption(100, 100, 1.5, True)
        np.testing.assert_allclose(bs.calculate_price(instrument), 13.806724108166861)

    def test_calculate_european_put_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = EuropeanOption(100, 100, 1.5, False)
        np.testing.assert_allclose(bs.calculate_price(instrument), 9.536519386171328)

    def test_calculate_cash_or_nothing_call_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = CashOrNothingBinaryOption(100, 100, 1.5, 100, True)
        np.testing.assert_allclose(bs.calculate_price(instrument), 46.16052683406444)

    def test_calculate_cash_or_nothing_put_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = CashOrNothingBinaryOption(100, 100, 1.5, 100, False)
        np.testing.assert_allclose(bs.calculate_price(instrument), 46.61382179879085)

    def test_calculate_asset_or_nothing_call_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = AssetOrNothingBinaryOption(100, 100, 1.5, True)
        np.testing.assert_allclose(bs.calculate_price(instrument), 59.9672509422313)

    def test_calculate_asset_or_nothing_put_price(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = AssetOrNothingBinaryOption(100, 100, 1.5, False)
        np.testing.assert_allclose(bs.calculate_price(instrument), 37.07730241261952)

    # Delta calculation test

    def test_calculate_forward_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = Forward(100, 104.6027859908717, 1.5)
        np.testing.assert_allclose(bs.calculate_delta(instrument), 1.046027859908717)

    def test_calculate_european_call_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = EuropeanOption(100, 100, 1.5, True)
        np.testing.assert_allclose(bs.calculate_delta(instrument), 0.599672509422313)

    def test_calculate_european_put_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = EuropeanOption(100, 100, 1.5, False)
        np.testing.assert_allclose(bs.calculate_delta(instrument), -0.37077302412619517)

    def test_calculate_cash_or_nothing_call_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = CashOrNothingBinaryOption(100, 100, 1.5, 100, True)
        np.testing.assert_allclose(bs.calculate_delta(instrument), 1.2087714628471327)

    def test_calculate_cash_or_nothing_put_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = CashOrNothingBinaryOption(100, 100, 1.5, 100, False)
        np.testing.assert_allclose(bs.calculate_delta(instrument), -1.2087714628471327)

    def test_calculate_asset_or_nothing_call_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = AssetOrNothingBinaryOption(100, 100, 1.5, True)
        np.testing.assert_allclose(bs.calculate_delta(instrument), 1.808443972269446)

    def test_calculate_asset_or_nothing_put_delta(self):
        bs = BlackScholes(0.05, 0.25, 0.02)
        instrument = AssetOrNothingBinaryOption(100, 100, 1.5, False)
        np.testing.assert_allclose(bs.calculate_delta(instrument), -0.8379984387209378)
