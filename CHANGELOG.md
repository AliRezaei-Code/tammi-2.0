# Changelog

All notable changes to TAMMI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-12-02

### Added (CLI Tool - Optional Enhancement)

> **Note**: These additions are optional and do not replace the original Jupyter notebook. The notebook continues to work exactly as before.

#### New Command-Line Interface
- `tammi_cli.py` - Process large batches of text files without Jupyter
- Interactive menu mode (`--menu`) for users who prefer guided input
- Progress bar with time estimates for long-running analyses

#### New Package Structure (`tammi/`)
- Modular Python package for developers who want to integrate TAMMI
- `tammi.analysis` - Core morphological analysis logic
- `tammi.cli` - Command-line interface components  
- `tammi.io` - Input/output handlers for various formats
- `tammi.tests` - Unit tests (26 tests, all passing)

#### New Input/Output Formats
- **JSON support**: `--input-json` and `--output-json` flags
- **JSON Lines (JSONL)**: Streaming format for large datasets
- **MongoDB**: Database support alongside existing SQLite/MySQL/PostgreSQL

#### New Documentation (`docs/` folder)
- `user_guide.md` - Step-by-step installation and usage guide
- `api_reference.md` - Detailed API documentation for developers
- `architecture.md` - System design with diagrams
- `crc_cards.md` - Class-Responsibility-Collaboration cards
- `contributing.md` - How to contribute to the project

#### Package Configuration
- `pyproject.toml` - Modern Python packaging
- Optional dependencies: `[mysql]`, `[postgresql]`, `[mongodb]`, `[gpu]`

### Technical Details
- Supports Python 3.9+
- GPU acceleration via spaCy (CUDA and Apple Silicon MPS)
- Design patterns: Strategy Pattern for I/O, Factory Pattern for object creation

---

## Original Release

The original TAMMI implementation includes:
- `Tammi_simp_morpholex_batch_github.ipynb` - Jupyter notebook for morphological analysis
- `morpho_lex_df_w_log_w_prefsuf_no_head.csv` - MorphoLex dictionary
- 43 morphological metrics based on MorphoLex and MCI

---

## Contributors

- **TAMMI Original Contributors** - Jupyter notebook and initial implementation
- **Ali Rezaei** (ali0rezaei0@gmail.com) - CLI tool and documentation
