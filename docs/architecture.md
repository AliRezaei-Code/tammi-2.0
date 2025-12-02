# TAMMI Architecture Overview

## Design Philosophy

TAMMI v2.0 was refactored with the following goals:

- **Modularity**: Separate concerns into distinct modules
- **Extensibility**: Easy to add new input/output formats
- **Testability**: Comprehensive unit test coverage
- **Maintainability**: Clear interfaces and documentation

## Architecture Diagram

```mermaid
flowchart TD
    subgraph CLI["CLI Layer"]
        main["tammi/cli/main.py<br/>Argument parsing"]
        menu["tammi/cli/menu.py<br/>Interactive menu"]
        runner["tammi/cli/runner.py<br/>Pipeline orchestration"]
        progress["tammi/cli/progress.py<br/>Progress bar"]
    end

    subgraph IO["I/O Module (Strategy)"]
        readers["InputReader implementations<br/>CSV/JSON/JSONL/Text/Database"]
        writers["OutputWriter implementations<br/>CSV/JSON/JSONL/Database"]
    end

    subgraph Analysis["Analysis Module"]
        analyzer["TAMMIAnalyzer<br/>spaCy + MorphoLex"]
        morpholex["MorphoLexDict"]
        metrics["metrics.py constants/helpers"]
    end

    main --> menu
    menu --> runner
    runner --> readers
    readers --> analyzer
    analyzer --> writers
    analyzer --> morpholex
    analyzer --> metrics
    runner --> progress
```

## Design Patterns Used

### 1. Strategy Pattern (I/O Module)

The Strategy Pattern allows interchangeable algorithms (I/O formats) at runtime.

```mermaid
classDiagram
    class InputReader {
        +stream() Iterator[Tuple[str,str]]
        +count() int
        +close() None
    }
    class CSVReader
    class JSONReader
    class JSONLReader
    class TextFileReader
    class DatabaseReader
    class SQLiteReader
    class MySQLReader
    class PostgreSQLReader
    class MongoDBReader

    InputReader <|-- CSVReader
    InputReader <|-- JSONReader
    InputReader <|-- JSONLReader
    InputReader <|-- TextFileReader
    InputReader <|-- DatabaseReader
    DatabaseReader <|-- SQLiteReader
    DatabaseReader <|-- MySQLReader
    DatabaseReader <|-- PostgreSQLReader
    DatabaseReader <|-- MongoDBReader
```

### 2. Factory Pattern (Reader/Writer Creation)

```mermaid
classDiagram
    class ReaderFactory {
        -_readers: Dict[str, Type[InputReader]]
        +register(type, reader_class)
        +create(type, **kwargs) InputReader
        +available_types() List[str]
    }
    class WriterFactory {
        -_writers: Dict[str, Type[OutputWriter]]
        +register(type, writer_class)
        +create(type, columns, **kwargs) OutputWriter
        +available_types() List[str]
    }
    ReaderFactory --> InputReader
    WriterFactory --> OutputWriter
```

### 3. Template Method Pattern (Analysis Pipeline)

```mermaid
sequenceDiagram
    participant Runner as TAMMIRunner
    participant Reader as InputReader
    participant Analyzer as TAMMIAnalyzer
    participant Writer as OutputWriter

    Runner->>Analyzer: init (load spaCy, MorphoLex)
    Runner->>Reader: stream()
    loop texts
        Reader-->>Runner: (text_id, text)
        Runner->>Analyzer: analyze_text()
        Analyzer-->>Runner: metrics
        Runner->>Writer: write_row()/buffer
    end
    Runner->>Writer: write_all()/flush
    Runner-->>Runner: return count
```

## Module Dependencies

```mermaid
flowchart TD
    tammi["tammi/__init__.py"] --> analysis_init["analysis/__init__.py"]
    tammi --> cli_init["cli/__init__.py"]
    tammi --> io_init["io/__init__.py"]

    subgraph Analysis
        analyzer["analyzer.py<br/>spaCy + metrics + morpholex"]
        morpholex["morpholex.py<br/>csv"]
        metrics_mod["metrics.py<br/>constants/helpers"]
    end

    subgraph CLI
        main_cli["main.py<br/>argparse + I/O factories + progress + menu + runner"]
        menu_cli["menu.py<br/>io.base + analysis.metrics"]
        progress_cli["progress.py<br/>spacy/time/os"]
        runner_cli["runner.py<br/>analysis.analyzer + io.base + progress"]
    end

    subgraph IO
        base_io["base.py<br/>abc/dataclasses/typing"]
        csv_io["csv_io.py<br/>csv + base"]
        json_io["json_io.py<br/>json + base"]
        file_io["file_io.py<br/>pathlib + base"]
        db_io["database_io.py<br/>sqlite3/mysql?/psycopg2?/pymongo?"]
    end

    analysis_init --> analyzer
    analysis_init --> morpholex
    analysis_init --> metrics_mod

    cli_init --> main_cli
    cli_init --> menu_cli
    cli_init --> progress_cli
    cli_init --> runner_cli

    io_init --> base_io
    io_init --> csv_io
    io_init --> json_io
    io_init --> file_io
    io_init --> db_io
```

## Data Flow

### Input Processing Flow

```mermaid
flowchart LR
    subgraph Sources
        files["Text files"]
        csv["CSV"]
        json["JSON"]
        jsonl["JSONL"]
        sqlite["SQLite"]
        mysql["MySQL"]
        pg["PostgreSQL"]
        mongo["MongoDB"]
    end

    subgraph Pipeline
        reader["InputReader.stream()"]
        analyzer["TAMMIAnalyzer.analyze_text()"]
        writer["OutputWriter"]
    end

    subgraph Destinations
        out_csv["CSV"]
        out_json["JSON"]
        out_db["Databases"]
    end

    files --> reader
    csv --> reader
    json --> reader
    jsonl --> reader
    sqlite --> reader
    mysql --> reader
    pg --> reader
    mongo --> reader

    reader --> analyzer
    analyzer --> writer
    writer --> out_csv
    writer --> out_json
    writer --> out_db
```

### Analysis Pipeline Detail

```mermaid
flowchart TD
    A["Input text"] --> B["1. Preprocessing<br/>Lowercase (optional), spaCy tokenize/POS"]
    B --> C["2. Content Word Extraction<br/>Filter NOUN/VERB/ADJ/ADV, lemmas"]
    C --> D["3. MorphoLex Lookup<br/>Prefix/root/suffix, freq/family size"]
    D --> E["4. Inflection Analysis<br/>Compare token vs lemma, variety/TTR"]
    E --> F["5. Derivation Analysis<br/>Affix detection, variety/TTR, MCI"]
    F --> G["6. Metric Aggregation<br/>Window means, normalization, package results"]
    G --> H["Results dictionary (43 metrics)"]
```

## Error Handling Strategy

```mermaid
flowchart TD
    cli_err["CLI Layer (main.py)<br/>SystemExit for user errors<br/>Path validation<br/>DB string parsing"]
    runner_err["Runner Layer (runner.py)<br/>FileNotFound for MorphoLex<br/>spaCy load guard<br/>Progress updates on errors"]
    io_err["I/O Layer (base.py, *_io.py)<br/>ImportError for optional deps<br/>ConnectionError for DB issues<br/>ValueError for bad formats"]
    analysis_err["Analysis Layer (analyzer.py)<br/>Safe divide helpers<br/>Missing dictionary entries handled<br/>Empty text handling"]

    cli_err --> runner_err --> io_err --> analysis_err
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
