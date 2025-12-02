"""
Analysis module for TAMMI morphological analysis.

Refactored into modular structure by Ali Rezaei.
"""

__author__ = "Ali Rezaei"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tammi.analysis.analyzer import TAMMIAnalyzer
    from tammi.analysis.morpholex import MorphoLexDict

from tammi.analysis.metrics import COLUMN_NAMES, DERIVATIONAL_AFFIX_INDICES

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "TAMMIAnalyzer":
        from tammi.analysis.analyzer import TAMMIAnalyzer
        return TAMMIAnalyzer
    elif name == "MorphoLexDict":
        from tammi.analysis.morpholex import MorphoLexDict
        return MorphoLexDict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TAMMIAnalyzer",
    "MorphoLexDict",
    "COLUMN_NAMES",
    "DERIVATIONAL_AFFIX_INDICES",
]
