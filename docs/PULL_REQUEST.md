# Pull Request: TAMMI v2.0.0 - Major Refactoring (local) 

## Summary

This PR introduces a major refactoring of TAMMI (Tool for Automatic Measurement of Morphological Information) from a monolithic script to a modular Python package with enhanced features and comprehensive documentation.

## Changes

### New Package Structure

Refactored `tammi_cli.py` into a proper Python package:

```text
tammi/
├── __init__.py              # Package root with lazy imports
├── analysis/                # Morphological analysis module
│   ├── analyzer.py          # TAMMIAnalyzer class
│   ├── metrics.py           # Constants and helper functions
│   └── morpholex.py         # MorphoLex dictionary loader
├── cli/                     # Command-line interface
│   ├── main.py              # Entry point and argument parsing
│   ├── menu.py              # Interactive menu system
│   ├── progress.py          # Progress bar utilities
│   └── runner.py            # Pipeline orchestrator
├── io/                      # I/O module (Strategy Pattern)
│   ├── base.py              # Abstract classes and factories
│   ├── csv_io.py            # CSV reader/writer
│   ├── json_io.py           # JSON/JSONL reader/writer
│   ├── file_io.py           # Text file reader
│   └── database_io.py       # Database readers/writers
└── tests/                   # Unit tests (26 tests)
```

### Design Patterns

- **Strategy Pattern**: `InputReader` and `OutputWriter` abstract base classes for interchangeable I/O formats
- **Factory Pattern**: `ReaderFactory` and `WriterFactory` for dynamic object creation

### New Features

1. **JSON/JSONL Support**
   - `--input-json` flag for JSON input files
   - `--output-json` flag for JSON output
   - Support for array and object JSON formats
   - JSON Lines (JSONL) support for streaming

2. **MongoDB Support**
   - New `MongoDBReader` and `MongoDBWriter` classes
   - Connection string format: `mongodb://user:pass@host/db`

3. **Enhanced CLI**
   - Interactive menu with numbered choices (`--menu`)
   - Better progress bar with ETA
   - Auto-detected batch size and process count

4. **Modern Packaging**
   - `pyproject.toml` configuration
   - Optional dependencies: `[mysql]`, `[postgresql]`, `[mongodb]`, `[gpu]`, `[all]`
   - Entry point: `tammi` command

### Documentation

New `docs/` folder with comprehensive documentation:

- `user_guide.md` - Installation and usage
- `api_reference.md` - Detailed API docs
- `architecture.md` - System design with diagrams
- `crc_cards.md` - Class responsibilities
- `contributing.md` - Contribution guidelines

### Testing

- 26 unit tests covering I/O and analysis modules
- All tests passing
- Integration tests for CLI workflows

## Breaking Changes

- The original `tammi_cli.py` has been replaced with a thin wrapper
- Original code preserved in `tammi_cli_legacy.py`

## Migration Guide

No changes needed for basic usage:

```bash
# Still works the same
python tammi_cli.py texts/ -o results.csv
```

New features available:

```bash
# JSON input/output
python tammi_cli.py --input-json data.json -o results.json --output-json

# Interactive menu
python tammi_cli.py --menu
```

## Testing Commands

```bash
# Run unit tests
python -m unittest discover tammi/tests/ -v

# Test CLI
python tammi_cli.py test_texts/ -o test_output.csv
```

## Checklist

- [x] Package structure created
- [x] Design patterns implemented (Strategy, Factory)
- [x] JSON/JSONL I/O support added
- [x] MongoDB support added
- [x] Interactive menu enhanced
- [x] Unit tests written and passing (26 tests)
- [x] Documentation complete
- [x] pyproject.toml configured
- [x] .gitignore updated
- [x] CHANGELOG.md created
- [x] README.md updated

## Files Changed

### New Files

- `tammi/` - Entire package (19 Python files)
- `docs/` - Documentation (6 markdown files)
- `pyproject.toml` - Package configuration
- `CHANGELOG.md` - Version history
- `tammi_cli_legacy.py` - Backup of original code

### Modified Files

- `tammi_cli.py` - Now a thin wrapper
- `README.md` - Updated with new features
- `.gitignore` - Comprehensive exclusions

---

**Author**: Ali Rezaei (<ali0rezaei0@gmail.com>)
