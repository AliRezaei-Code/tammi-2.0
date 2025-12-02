"""
TAMMI - Tool for Automatic Measurement of Morphological Information

A Python package for analyzing morphological complexity in text using 
spaCy and the MorphoLex dictionary.

Package refactored and enhanced by Ali Rezaei:
- Modular package structure with Strategy and Factory design patterns
- JSON/JSONL input/output support
- MongoDB database support (alongside SQLite/MySQL/PostgreSQL)
- Enhanced CLI with interactive menu
- Comprehensive unit tests
"""

__version__ = "2.0.0"
__author__ = "TAMMI Original Contributors"
__maintainer__ = "Ali Rezaei"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tammi.analysis.analyzer import TAMMIAnalyzer
    from tammi.analysis.morpholex import MorphoLexDict

# Lazy imports to avoid circular dependencies
# Note: Pylance may show warnings about __all__ entries not being present,
# but this is a known limitation with lazy imports - they work at runtime.
def __getattr__(name: str):
    if name == "TAMMIAnalyzer":
        from tammi.analysis.analyzer import TAMMIAnalyzer
        return TAMMIAnalyzer
    elif name == "MorphoLexDict":
        from tammi.analysis.morpholex import MorphoLexDict
        return MorphoLexDict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TAMMIAnalyzer", "MorphoLexDict", "__version__"]  # noqa: F822
