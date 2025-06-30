import numpy as np
import pytest

from source.models.black_scholes import BlackScholes
from source.instruments.european_option import EuropeanOption


class TestBlackScholes:
    # Initialization Tests
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2)
        assert bs.r == 0.02
        assert bs.sigma == 0.2

    # support_instrument Method Test
    def test_instrument_support(self):
        """Test supports_instrument method produces expected output"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2)
        euro_option = EuropeanOption(100, True, underlying=100, maturity=1)
        assert bs.supports_instrument(euro_option) == True

    # Validation Method Test
    def test_validate_inputs(self):
        """Test validation_inputs method produces expected output"""
        bs = BlackScholes(risk_free_rate=0.02, volatility=0.2)
        with pytest.raises(ValueError):
            bs.validate_inputs("Not Supported Instrument")

    # Parameter Validation Tests
    def test_negative_risk_free_rate_raises_error(self):
        """Test that negative risk free rate raises error"""
        with pytest.raises(ValueError):
            BlackScholes(-0.02, 0.2)

    def test_zero_volatility_raises_error(self):
        """Test that zero volatility raises error"""
        with pytest.raises(ValueError):
            BlackScholes(0.02, 0)

    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises error"""
        with pytest.raises(ValueError):
            BlackScholes(0.02, -0.2)

    # Price calculation test
    def test_calculate_call_price(self):
        bs = BlackScholes(0.05, 0.25)
        instrument = EuropeanOption(100, True, underlying=100, maturity=1)
        np.testing.assert_allclose(bs.calculate_price(instrument), 12.336, atol=1e-3)

    def test_calculate_put_price(self):
        bs = BlackScholes(0.05, 0.25)
        instrument = EuropeanOption(100, False, underlying=100, maturity=1)
        np.testing.assert_allclose(bs.calculate_price(instrument), 7.458, atol=1e-3)
