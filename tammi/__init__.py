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

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "TAMMIAnalyzer":
        from tammi.analysis.analyzer import TAMMIAnalyzer
        return TAMMIAnalyzer
    elif name == "MorphoLexDict":
        from tammi.analysis.morpholex import MorphoLexDict
        return MorphoLexDict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TAMMIAnalyzer", "MorphoLexDict", "__version__"]
