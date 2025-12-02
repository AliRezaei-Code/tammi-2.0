# Contributing to TAMMI

Thank you for your interest in contributing to TAMMI! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Adding New Features](#adding-new-features)

---

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- pip

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tammi.git
   cd tammi
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/tammi.git
   ```

---

## Development Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Project Structure

```
tammi/
├── tammi/                  # Main package
│   ├── __init__.py
│   ├── analysis/           # Analysis module
│   ├── cli/                # CLI module
│   ├── io/                 # I/O module
│   └── tests/              # Unit tests
├── docs/                   # Documentation
├── pyproject.toml          # Package configuration
├── README.md
└── tammi_cli.py            # CLI wrapper
```

---

## Code Style

### Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format all files
black tammi/

# Check formatting without changes
black --check tammi/
```

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
# Run linter
ruff check tammi/

# Auto-fix issues
ruff check --fix tammi/
```

### Type Hints

- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- For optional imports (mysql, psycopg2, etc.), use `# type: ignore[import-not-found]`

Example:
```python
from __future__ import annotations

def process_text(text: str, lowercase: bool = True) -> dict[str, Any]:
    """Process text and return metrics."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def analyze_text(text: str, text_id: str) -> dict[str, Any]:
    """
    Analyze a single text for morphological features.
    
    Args:
        text: The text to analyze.
        text_id: Unique identifier for the text.
        
    Returns:
        Dictionary containing morphological metrics.
        
    Raises:
        ValueError: If text is empty.
        
    Example:
        >>> analyzer = TAMMIAnalyzer("morpholex.csv")
        >>> result = analyzer.analyze_text("Hello world", "doc1")
    """
```

---

## Testing

### Running Tests

```bash
# Run all tests
python -m unittest discover tammi/tests/ -v

# Run specific test file
python -m unittest tammi/tests/test_io.py -v

# Run specific test class
python -m unittest tammi.tests.test_io.TestCSVIO -v

# With pytest (if installed)
pytest tammi/tests/ -v
```

### Writing Tests

Place tests in `tammi/tests/`. Follow the naming convention `test_*.py`.

Example test:
```python
import unittest
from tammi.io.csv_io import CSVReader

class TestCSVReader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_file = "test_data.csv"
        # Create test data...
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files...
    
    def test_stream_returns_tuples(self):
        """Test that stream yields (id, text) tuples."""
        reader = CSVReader(self.test_file)
        for item in reader.stream():
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
        reader.close()
    
    def test_count_matches_actual(self):
        """Test that count returns correct number."""
        reader = CSVReader(self.test_file)
        count = reader.count()
        actual = sum(1 for _ in reader.stream())
        self.assertEqual(count, actual)
        reader.close()
```

### Test Coverage

```bash
# Run with coverage
pytest tammi/tests/ --cov=tammi --cov-report=html

# View report
open htmlcov/index.html
```

---

## Pull Request Process

### Before Submitting

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Run tests:**
   ```bash
   python -m unittest discover tammi/tests/ -v
   ```

4. **Format code:**
   ```bash
   black tammi/
   ruff check --fix tammi/
   ```

5. **Update documentation** if needed

6. **Commit with clear message:**
   ```bash
   git add .
   git commit -m "Add feature: description of what you added"
   ```

### Submitting

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill in the PR template:
   - Describe what changes you made
   - Link any related issues
   - Note any breaking changes

### PR Review

- Address reviewer feedback
- Keep commits clean (squash if needed)
- Ensure CI passes

---

## Adding New Features

### Adding a New Input Format

1. Create `tammi/io/newformat_io.py`:

```python
from tammi.io.base import InputReader, ReaderFactory

class NewFormatReader(InputReader):
    def __init__(self, path: str, text_column: str, id_column: str, lowercase: bool = True):
        self.path = path
        self.text_column = text_column
        self.id_column = id_column
        self.lowercase = lowercase
        self._data = self._load_data()
    
    def _load_data(self):
        # Load your format here
        pass
    
    def stream(self):
        for item in self._data:
            text_id = item[self.id_column]
            text = item[self.text_column]
            if self.lowercase:
                text = text.lower()
            yield (text_id, text)
    
    def count(self):
        return len(self._data)
    
    def close(self):
        pass

# Register with factory
ReaderFactory.register("newformat", NewFormatReader)
```

2. Add to `tammi/io/__init__.py`:
```python
from tammi.io.newformat_io import NewFormatReader
```

3. Add CLI argument in `tammi/cli/main.py`:
```python
parser.add_argument("--input-newformat", metavar="PATH", help="...")
```

4. Write tests in `tammi/tests/test_io.py`

5. Update documentation

### Adding New Metrics

1. Add column name to `COLUMN_NAMES` in `tammi/analysis/metrics.py`

2. Implement calculation in `TAMMIAnalyzer.analyze_text()`

3. Update tests

4. Update documentation

---

## Questions?

- Open an issue on GitHub
- Contact the maintainers

Thank you for contributing!
