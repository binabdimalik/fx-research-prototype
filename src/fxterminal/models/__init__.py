"""SQLAlchemy models. Importing this package registers all tables on `Base`."""

from fxterminal.models.analysis import CurrencyScore, StanceScore, Theme, TradeIdea
from fxterminal.models.base import Base
from fxterminal.models.documents import RawDocument
from fxterminal.models.market import FxPrice, MacroEvent, RateExpectation
from fxterminal.models.ops import ProviderRun

__all__ = [
    "Base",
    "CurrencyScore",
    "FxPrice",
    "MacroEvent",
    "ProviderRun",
    "RateExpectation",
    "RawDocument",
    "StanceScore",
    "Theme",
    "TradeIdea",
]
