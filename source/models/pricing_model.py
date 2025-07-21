from abc import ABC, abstractmethod
from ..instruments import derivative_instrument as d


class PricingModel(ABC):
    @abstractmethod
    def supports_instrument(self, instrument: d.DerivativeInstrument) -> bool:
        """Check if model can price this instrument type"""
        pass

    @abstractmethod
    def calculate_price(self, instrument: d.DerivativeInstrument) -> float:
        """Main pricing method"""
        pass

    @abstractmethod
    def calculate_delta(self, instrument: d.DerivativeInstrument) -> float:
        """Main delta method"""
        pass

    @abstractmethod
    def calculate_vega(self, instrument: d.DerivativeInstrument) -> float:
        """Main vega method"""
        pass

    @abstractmethod
    def calculate_theta(self, instrument: d.DerivativeInstrument) -> float:
        """Main theta method"""
        pass

    @abstractmethod
    def calculate_rho(self, instrument: d.DerivativeInstrument) -> float:
        """Main rho method"""
        pass

    @abstractmethod
    def calculate_gamma(self, instrument: d.DerivativeInstrument) -> float:
        """Main gamma method"""
        pass

    def validate_inputs(self, instrument: d.DerivativeInstrument):
        """Common validation for all models"""
        if not self.supports_instrument(instrument):
            raise ValueError(
                f"{self.__class__.__name__} does not support "
                f"instrument type {instrument.__class__.__name__}"
            )
