# TAMMI Documentation

Welcome to the TAMMI (Tool for Automatic Measurement of Morphological Information) documentation.

## Contents

1. [**Architecture Overview**](architecture.md) - System design, patterns, and component interactions
2. [**API Reference**](api_reference.md) - Detailed documentation of all modules and classes
3. [**User Guide**](user_guide.md) - How to install and use TAMMI
4. [**CRC Cards**](crc_cards.md) - Class-Responsibility-Collaboration diagrams
5. [**Contributing**](contributing.md) - How to contribute to TAMMI

## Quick Links

- [Installation Guide](user_guide.md#installation)
- [CLI Usage Examples](user_guide.md#cli-usage)
- [Python API Examples](user_guide.md#python-api)
- [Database Support](user_guide.md#database-support)

## Package Structure

```
tammi/
├── __init__.py          # Package root with lazy imports
├── analysis/            # Morphological analysis module
│   ├── __init__.py
│   ├── analyzer.py      # TAMMIAnalyzer class
│   ├── metrics.py       # Constants and helper functions
│   └── morpholex.py     # MorphoLex dictionary loader
├── cli/                 # Command-line interface module
│   ├── __init__.py
│   ├── main.py          # CLI entry point and argument parsing
│   ├── menu.py          # Interactive menu system
│   ├── progress.py      # Progress bar and system utilities
│   └── runner.py        # TAMMIRunner orchestrator
├── io/                  # Input/Output module (Strategy Pattern)
│   ├── __init__.py
│   ├── base.py          # Abstract base classes and factories
│   ├── csv_io.py        # CSV reader/writer
│   ├── json_io.py       # JSON/JSONL reader/writer
│   ├── file_io.py       # Text file reader
│   └── database_io.py   # Database readers/writers
└── tests/               # Unit tests
    ├── __init__.py
    ├── test_io.py
    └── test_analysis.py
```

## Version History

- **v2.0.0** (December 2024) - Major refactor by Ali Rezaei
  - Modular package structure with design patterns
  - JSON/JSONL input/output support
  - MongoDB database support
  - Enhanced CLI with interactive menu
  - Comprehensive unit tests
  
- **v1.0.0** - Original TAMMI implementation
