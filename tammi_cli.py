#!/usr/bin/env python3
"""
TAMMI CLI - Command-line interface for morphological analysis.

Tool for Automatic Measurement of Morphological Information (TAMMI)
Refactored and enhanced by Ali Rezaei

This is a simple wrapper that imports the main CLI from the tammi package.
For the full implementation, see tammi/cli/main.py.

Features:
- Text file, CSV, JSON/JSONL input support
- CSV and JSON output formats
- Database support: SQLite, MySQL, PostgreSQL, MongoDB
- GPU acceleration (CUDA/MPS)
- Interactive menu mode

Usage:
    python tammi_cli.py responses/ -o results.csv
    python tammi_cli.py --input-csv data.csv --text-column text -o results.csv
    python tammi_cli.py --input-json data.json --text-column text -o results.json
    python tammi_cli.py --menu
"""

__author__ = "Ali Rezaei"
__version__ = "2.0.0"

from tammi.cli.main import main

if __name__ == "__main__":
    main()
