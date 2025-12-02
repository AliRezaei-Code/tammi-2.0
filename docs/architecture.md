# TAMMI Architecture Overview

## Design Philosophy

TAMMI v2.0 was refactored with the following goals:

- **Modularity**: Separate concerns into distinct modules
- **Extensibility**: Easy to add new input/output formats
- **Testability**: Comprehensive unit test coverage
- **Maintainability**: Clear interfaces and documentation

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TAMMI CLI (main.py)                            │
│                         Command-line argument parsing                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Interactive Menu (menu.py)                         │
│                    Numbered choice menu for user interaction                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TAMMIRunner (runner.py)                           │
│                         Orchestrates analysis pipeline                        │
│                                                                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  InputReader    │───▶│  TAMMIAnalyzer  │───▶│  OutputWriter   │          │
│  │  (Strategy)     │    │  (Analysis)     │    │  (Strategy)     │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    I/O Module       │   │   Analysis Module   │   │    CLI Module       │
│                     │   │                     │   │                     │
│ • CSVReader/Writer  │   │ • TAMMIAnalyzer     │   │ • ProgressBar       │
│ • JSONReader/Writer │   │ • MorphoLexDict     │   │ • interactive_menu  │
│ • TextFileReader    │   │ • Metrics           │   │ • quick_cpu_probe   │
│ • DatabaseReaders   │   │                     │   │ • check_gpu         │
│ • DatabaseWriters   │   │                     │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

## Design Patterns Used

### 1. Strategy Pattern (I/O Module)

The Strategy Pattern allows interchangeable algorithms (I/O formats) at runtime.

```text
┌──────────────────────────────────────────────────────────────┐
│                    <<abstract>>                               │
│                    InputReader                                │
├──────────────────────────────────────────────────────────────┤
│ + stream() -> Iterator[Tuple[str, str]]                      │
│ + count() -> int                                             │
│ + close() -> None                                            │
└──────────────────────────────────────────────────────────────┘
                              △
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CSVReader   │    │   JSONReader    │    │ TextFileReader  │
└──────────────┘    └─────────────────┘    └─────────────────┘
       │                      │                      │
       │              ┌───────┴───────┐              │
       │              │               │              │
       │       ┌──────────┐    ┌───────────┐        │
       │       │JSONLReader│   │DatabaseReader│     │
       │       └──────────┘    └───────────┘        │
       │                              │              │
       │              ┌───────────────┼──────────────┐
       │              │               │              │
       │       ┌──────────┐    ┌───────────┐  ┌───────────┐
       │       │SQLiteReader│  │MySQLReader│  │MongoDBReader│
       │       └──────────┘    └───────────┘  └───────────┘
```

### 2. Factory Pattern (Reader/Writer Creation)

The Factory Pattern encapsulates object creation logic.

```python
# Example: Creating a reader based on type
reader = ReaderFactory.create("csv", path="data.csv", text_column="text")
reader = ReaderFactory.create("json", path="data.json", text_column="content")
reader = ReaderFactory.create("sqlite", db_config=config)
```

```text
┌────────────────────────────────────────────────────────────────┐
│                      ReaderFactory                              │
├────────────────────────────────────────────────────────────────┤
│ - _readers: Dict[str, Type[InputReader]]                       │
├────────────────────────────────────────────────────────────────┤
│ + register(type: str, reader_class: Type[InputReader])         │
│ + create(type: str, **kwargs) -> InputReader                   │
│ + available_types() -> List[str]                               │
└────────────────────────────────────────────────────────────────┘
```

### 3. Template Method Pattern (Analysis Pipeline)

The `TAMMIRunner.run()` method defines the skeleton of the analysis algorithm.

```text
┌─────────────────────────────────────────────────────────────────┐
│                       TAMMIRunner.run()                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Initialize analyzer (load spaCy, MorphoLex)                 │
│  2. Read texts from InputReader.stream()                        │
│  3. Process texts through TAMMIAnalyzer.analyze_text()          │
│  4. Collect results                                              │
│  5. Write results via OutputWriter.write_all()                  │
│  6. Return count of processed items                              │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```text
tammi/
├── __init__.py
│   └── imports: analysis.analyzer, analysis.morpholex
│
├── analysis/
│   ├── __init__.py
│   │   └── imports: metrics, analyzer, morpholex
│   ├── analyzer.py
│   │   └── imports: spacy, metrics, morpholex
│   ├── metrics.py
│   │   └── imports: (none - constants and pure functions)
│   └── morpholex.py
│       └── imports: csv (stdlib)
│
├── cli/
│   ├── __init__.py
│   │   └── imports: main, menu, runner, progress
│   ├── main.py
│   │   └── imports: argparse, io.base, io.database_io, 
│   │                cli.progress, cli.menu, cli.runner
│   ├── menu.py
│   │   └── imports: io.base, analysis.metrics
│   ├── progress.py
│   │   └── imports: spacy, time, os
│   └── runner.py
│       └── imports: analysis.analyzer, io.base, cli.progress
│
└── io/
    ├── __init__.py
    │   └── imports: base, csv_io, json_io, file_io, database_io
    ├── base.py
    │   └── imports: abc, dataclasses, typing
    ├── csv_io.py
    │   └── imports: csv, base
    ├── json_io.py
    │   └── imports: json, base
    ├── file_io.py
    │   └── imports: pathlib, base
    └── database_io.py
        └── imports: base, sqlite3, mysql.connector?, 
                     psycopg2?, pymongo?
```

## Data Flow

### Input Processing Flow

```text
Text Input Sources          Processing                    Output Destinations
─────────────────          ──────────                    ───────────────────

┌─────────────┐
│ Text Files  │──┐
└─────────────┘  │
                 │
┌─────────────┐  │    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
│  CSV File   │──┼───▶│ InputReader  │───▶│ TAMMIAnalyzer │───▶│   CSV    │
└─────────────┘  │    │   .stream()  │    │ .analyze_text │    └──────────┘
                 │    └──────────────┘    └───────────────┘           │
┌─────────────┐  │                               │                     │
│  JSON File  │──┤                               ▼                     ▼
└─────────────┘  │    ┌──────────────────────────────────────┐  ┌──────────┐
                 │    │           Analysis Results            │  │   JSON   │
┌─────────────┐  │    │  • Inflected tokens count            │  └──────────┘
│   JSONL     │──┤    │  • Derivational tokens count         │         │
└─────────────┘  │    │  • MorphoLex metrics                 │         │
                 │    │  • MCI (inflectional/derivational)   │         ▼
┌─────────────┐  │    │  • TTR metrics                       │  ┌──────────┐
│   SQLite    │──┤    └──────────────────────────────────────┘  │ Database │
└─────────────┘  │                                               │(SQLite/  │
                 │                                               │ MySQL/   │
┌─────────────┐  │                                               │ Postgres/│
│   MySQL     │──┤                                               │ MongoDB) │
└─────────────┘  │                                               └──────────┘
                 │
┌─────────────┐  │
│ PostgreSQL  │──┤
└─────────────┘  │
                 │
┌─────────────┐  │
│   MongoDB   │──┘
└─────────────┘
```

### Analysis Pipeline Detail

```text
Input Text
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                       1. Preprocessing                         │
│  • Lowercase text (optional)                                   │
│  • spaCy tokenization and POS tagging                          │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                    2. Content Word Extraction                  │
│  • Filter for NOUN, VERB, ADJ, ADV                            │
│  • Extract lemmas                                              │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                    3. MorphoLex Lookup                         │
│  • Match tokens against MorphoLex dictionary                  │
│  • Extract morpheme features (prefix, root, suffix)           │
│  • Get frequency and family size data                          │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                    4. Inflection Analysis                      │
│  • Compare token to lemma                                      │
│  • Identify inflectional morphemes                             │
│  • Calculate variety and TTR                                   │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                    5. Derivation Analysis                      │
│  • Identify derivational affixes                               │
│  • Calculate variety and TTR                                   │
│  • Compute MCI scores                                          │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│                    6. Metric Aggregation                       │
│  • Compute means across windows                                │
│  • Normalize by content word count                             │
│  • Package results as dictionary                               │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
Results Dictionary (43 metrics)
```

## Error Handling Strategy

```text
┌────────────────────────────────────────────────────────────────┐
│                     Error Handling Layers                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLI Layer (main.py)                                           │
│  ├── SystemExit for user-facing errors                         │
│  ├── Validation of file paths                                  │
│  └── Database connection string parsing                        │
│                                                                 │
│  Runner Layer (runner.py)                                      │
│  ├── FileNotFoundError for missing MorphoLex                   │
│  ├── Graceful handling of spaCy loading errors                 │
│  └── Progress bar updates on errors                            │
│                                                                 │
│  I/O Layer (base.py, *_io.py)                                  │
│  ├── ImportError for optional dependencies                     │
│  ├── ConnectionError for database issues                       │
│  └── ValueError for invalid data formats                       │
│                                                                 │
│  Analysis Layer (analyzer.py)                                  │
│  ├── Safe division helpers (division by zero)                  │
│  ├── Missing dictionary entries handled gracefully             │
│  └── Empty text handling                                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Extension Points

### Adding a New Input Format

1. Create a new reader class implementing `InputReader`:

```python
# tammi/io/xml_io.py
from tammi.io.base import InputReader, ReaderFactory

class XMLReader(InputReader):
    def __init__(self, path: str, text_xpath: str, id_xpath: str, lowercase: bool = True):
        self.path = path
        self.text_xpath = text_xpath
        self.id_xpath = id_xpath
        self.lowercase = lowercase
    
    def stream(self) -> Iterator[Tuple[str, str]]:
        # Implementation here
        pass
    
    def count(self) -> int:
        # Implementation here
        pass
    
    def close(self) -> None:
        pass

# Register with factory
ReaderFactory.register("xml", XMLReader)
```

1. Add CLI argument in `main.py`:

```python
parser.add_argument("--input-xml", metavar="XML_PATH", help="...")
```

### Adding a New Output Format

Similar process - implement `OutputWriter` and register with `WriterFactory`.

### Adding New Metrics

1. Add constants to `metrics.py`
2. Implement calculation in `analyzer.py`
3. Add column name to `COLUMN_NAMES` list
