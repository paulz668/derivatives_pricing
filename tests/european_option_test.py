import numpy as np
import pytest

from source.instruments.european_option import *
from source.models.black_scholes import BlackScholes


class TesteuropeanOption:
    # Initialization Tests
    def test_european_option_initialization_params(self):
        """Test initialization with custom parameters"""
        instrument = EuropeanOption(
            underlying=100, strike=100, time_to_maturity=1, is_call=True
        )
        assert instrument.underlying == 100
        assert instrument.strike == 100
        assert instrument.time_to_maturity == 1
        assert instrument.is_call == True

    # Parameter Validation Tests
    def test_negative_strike_raises_error(self):
        """Test negative strike raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, -100, 1, True)

    def test_zero_underlying_raises_error(self):
        """Test 0 underlying raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(0, 100, 1, True)

    def test_negative_underlying_raises_error(self):
        """Test negative underlying raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(-100, 100, 1, True)

    def test_zero_maturity_raises_error(self):
        """Test zero maturity raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, 100, 0, True)

    def test_negative_maturity_raises_error(self):
        """Test negative maturity raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, 100, -1, True)

    # Payoff Method Test
    def test_european_call_payoff(self):
        """Test payoff method produces expected output for a european call"""
        instrument = EuropeanOption(
            underlying=100, strike=100, time_to_maturity=1, is_call=True
        )
        underlying_values = 100 + np.arange(start=-3, stop=4)
        np.testing.assert_array_equal(
            instrument.payoff(underlying_values), np.array([0, 0, 0, 0, 1, 2, 3])
        )

    def test_european_put_payoff(self):
        """Test payoff method produces expected output for a european put"""
        instrument = EuropeanOption(
            underlying=100, strike=100, time_to_maturity=1, is_call=False
        )
        underlying_values = 100 + np.arange(start=-3, stop=4)
        np.testing.assert_array_equal(
            instrument.payoff(underlying_values), np.array([3, 2, 1, 0, 0, 0, 0])
        )

    # Price Method Test
    def test_price(self):
        """Test calculate_price produces expected output using Black Scholes model"""
        instrument = EuropeanOption(
            underlying=100, strike=100, time_to_maturity=1.5, is_call=True
        )
        model = BlackScholes(0.05, 0.25, 0.02)
        np.testing.assert_allclose(instrument.price(model), 13.806724108166861)
