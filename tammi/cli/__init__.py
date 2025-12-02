"""
CLI module for TAMMI.

Refactored and enhanced by Ali Rezaei with:
- Interactive menu with numbered choices
- JSON input/output support
- Database connectivity options
"""

__author__ = "Ali Rezaei"

from tammi.cli.main import main
from tammi.cli.menu import interactive_menu
from tammi.cli.runner import TAMMIRunner
from tammi.cli.progress import ProgressBar

__all__ = ["main", "interactive_menu", "TAMMIRunner", "ProgressBar"]
