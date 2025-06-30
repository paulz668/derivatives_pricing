import numpy as np
import pytest

from source.instruments.vanilla_options import *
from source.models.black_scholes import BlackScholes


class TesteuropeanOption:
    # Initialization Tests
    def test_european_option_initialization_params(self):
        """Test initialization with custom parameters"""
        instrument = EuropeanOption(100, True, underlying=100, maturity=1)
        assert instrument.strike == 100
        assert instrument.is_call == True
        assert instrument.underlying == 100
        assert instrument.maturity == 1

    # Parameter Validation Tests
    def test_negative_strike_raises_error(self):
        """Test negative strike raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(-5, True, underlying=100, maturity=1)

    def test_zero_underlying_raises_error(self):
        """Test 0 underlying raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, True, underlying=0, maturity=1)

    def test_negative_underlying_raises_error(self):
        """Test negative underlying raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, True, underlying=-100, maturity=1)

    def test_zero_maturity_raises_error(self):
        """Test zero maturity raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, True, underlying=100, maturity=0)

    def test_negative_maturity_raises_error(self):
        """Test negative maturity raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(100, True, underlying=100, maturity=-1)

    def test_not_none_payoff_fn_raises_error(self):
        """Test not None payoff_fn raises error"""
        with pytest.raises(ValueError):
            EuropeanOption(
                100,
                True,
                underlying=100,
                maturity=0,
                payoff_fn=lambda x: np.maximum(x, 0),
            )

    # Payoff Method Test
    def test_european_call_payoff(self):
        """Test payoff method produces expected output for a european call"""
        instrument = EuropeanOption(100, True, underlying=100, maturity=1)
        underlying_values = 100 + np.arange(start=-3, stop=4)
        np.testing.assert_array_equal(
            instrument.payoff(underlying_values), np.array([0, 0, 0, 0, 1, 2, 3])
        )

    def test_european_put_payoff(self):
        """Test payoff method produces expected output for a european put"""
        instrument = EuropeanOption(100, False, underlying=100, maturity=1)
        underlying_values = 100 + np.arange(start=-3, stop=4)
        np.testing.assert_array_equal(
            instrument.payoff(underlying_values), np.array([3, 2, 1, 0, 0, 0, 0])
        )

    # Price Method Test
    def test_price(self):
        """Test calculate_price produces expected output using Black Scholes model"""
        instrument = EuropeanOption(100, True, underlying=100, maturity=1)
        model = BlackScholes(0.05, 0.25)
        np.testing.assert_allclose(instrument.price(model), 12.336, atol=1e-3)
