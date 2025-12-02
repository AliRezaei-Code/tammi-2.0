# TAMMI API Reference

## Package: tammi

### tammi

```python
import tammi

tammi.__version__  # "2.0.0"
tammi.__author__   # "TAMMI Original Contributors"
tammi.__maintainer__  # "Ali Rezaei"
```

**Exports:**
- `TAMMIAnalyzer` - Main analysis class
- `MorphoLexDict` - MorphoLex dictionary loader

---

## Module: tammi.analysis

### tammi.analysis.analyzer

#### class TAMMIAnalyzer

Main class for performing morphological analysis on text.

```python
class TAMMIAnalyzer:
    def __init__(
        self,
        morpholex_path: str,
        spacy_model: str = "en_core_web_sm",
        use_gpu: bool = False,
        batch_size: int = 1000,
        n_process: int = 1,
        lowercase: bool = True
    ):
        """
        Initialize the TAMMI analyzer.
        
        Args:
            morpholex_path: Path to MorphoLex CSV file
            spacy_model: Name of spaCy model to use
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for spaCy pipe
            n_process: Number of processes for spaCy pipe
            lowercase: Whether to lowercase input text
        """
```

**Methods:**

```python
def analyze_text(self, text: str, text_id: str) -> Dict[str, Any]:
    """
    Analyze a single text and return morphological metrics.
    
    Args:
        text: The text to analyze
        text_id: Identifier for the text
        
    Returns:
        Dictionary with 43 morphological metrics
    """

def analyze_texts(
    self, 
    texts: Iterable[Tuple[str, str]]
) -> Iterator[Dict[str, Any]]:
    """
    Analyze multiple texts using spaCy pipe for efficiency.
    
    Args:
        texts: Iterable of (text_id, text_content) tuples
        
    Yields:
        Dictionary with metrics for each text
    """
```

**Example:**

```python
from tammi.analysis.analyzer import TAMMIAnalyzer

analyzer = TAMMIAnalyzer("morpho_lex.csv", use_gpu=True)
results = analyzer.analyze_text("The dogs are running.", "doc1")
print(results["Inflected_Tokens"])  # 0.5
```

---

### tammi.analysis.morpholex

#### class MorphoLexDict

Dictionary wrapper for MorphoLex data.

```python
class MorphoLexDict:
    def __init__(self, path: str):
        """
        Load MorphoLex CSV file.
        
        Args:
            path: Path to CSV file (without header)
        """
    
    def __getitem__(self, word: str) -> Optional[List[str]]:
        """Get morpheme data for a word."""
    
    def __contains__(self, word: str) -> bool:
        """Check if word exists in dictionary."""
    
    def __len__(self) -> int:
        """Return number of entries."""
    
    def get(self, word: str, default: Any = None) -> Optional[List[str]]:
        """Get morpheme data with default value."""
    
    def get_values(self, word: str, indices: List[int]) -> List[Optional[float]]:
        """
        Get specific values by column indices.
        
        Args:
            word: The word to look up
            indices: List of column indices to extract
            
        Returns:
            List of values (float or None if missing)
        """
```

**Example:**

```python
from tammi.analysis.morpholex import MorphoLexDict

morpholex = MorphoLexDict("morpho_lex.csv")
print(len(morpholex))  # ~70000

if "running" in morpholex:
    data = morpholex["running"]
    print(data)  # ['running', 'run', 'ning', ...]
```

---

### tammi.analysis.metrics

#### Constants

```python
COLUMN_NAMES: List[str]
# List of 43 output column names

DERIVATIONAL_AFFIX_INDICES: List[int]  
# Column indices for derivational affix data in MorphoLex
```

#### Functions

```python
def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, returning 0.0 if denominator is 0.
    """

def calc_mci(variety: float, num_subsets: int) -> float:
    """
    Calculate Morphological Complexity Index.
    
    Args:
        variety: Mean subset variety score
        num_subsets: Number of subsets
        
    Returns:
        MCI score
    """
```

---

## Module: tammi.io

### tammi.io.base

#### class InputReader (ABC)

Abstract base class for all input readers.

```python
class InputReader(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Tuple[str, str]]:
        """
        Stream (text_id, text_content) tuples.
        
        Yields:
            Tuple of (text_id, text_content)
        """
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of items."""
    
    def close(self) -> None:
        """Clean up resources."""
```

#### class OutputWriter (ABC)

Abstract base class for all output writers.

```python
class OutputWriter(ABC):
    @abstractmethod
    def write_row(self, row: Dict[str, Any]) -> None:
        """Write a single result row."""
    
    def write_batch(self, rows: List[Dict[str, Any]]) -> None:
        """Write multiple rows (default: calls write_row for each)."""
    
    @abstractmethod
    def write_all(self, rows: List[Dict[str, Any]]) -> None:
        """Write all results at once."""
    
    def close(self) -> None:
        """Clean up resources."""
```

#### class ReaderFactory

Factory for creating input readers.

```python
class ReaderFactory:
    @classmethod
    def register(cls, type_name: str, reader_class: Type[InputReader]) -> None:
        """Register a reader class for a type name."""
    
    @classmethod
    def create(cls, type_name: str, **kwargs) -> InputReader:
        """
        Create a reader instance.
        
        Args:
            type_name: Reader type ('csv', 'json', 'jsonl', 'files', 'sqlite', etc.)
            **kwargs: Arguments passed to reader constructor
            
        Returns:
            InputReader instance
        """
    
    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of registered reader types."""
```

#### class WriterFactory

Factory for creating output writers.

```python
class WriterFactory:
    @classmethod
    def register(cls, type_name: str, writer_class: Type[OutputWriter]) -> None:
        """Register a writer class for a type name."""
    
    @classmethod
    def create(cls, type_name: str, columns: List[str], **kwargs) -> OutputWriter:
        """
        Create a writer instance.
        
        Args:
            type_name: Writer type ('csv', 'json', 'jsonl', 'sqlite', etc.)
            columns: List of column names for output
            **kwargs: Arguments passed to writer constructor
            
        Returns:
            OutputWriter instance
        """
    
    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of registered writer types."""
```

#### check_available_drivers()

```python
def check_available_drivers() -> Dict[str, bool]:
    """
    Check which database/format drivers are available.
    
    Returns:
        Dictionary mapping driver names to availability (True/False)
    """
```

---

### tammi.io.csv_io

#### class CSVReader

```python
class CSVReader(InputReader):
    def __init__(
        self,
        path: str,
        text_column: str = "text_content",
        id_column: str = "text_id",
        lowercase: bool = True
    ):
        """
        Read from a CSV file.
        
        Args:
            path: Path to CSV file
            text_column: Name of column containing text
            id_column: Name of column containing IDs
            lowercase: Whether to lowercase text
        """
```

#### class CSVWriter

```python
class CSVWriter(OutputWriter):
    def __init__(self, path: str, columns: List[str]):
        """
        Write to a CSV file.
        
        Args:
            path: Output file path
            columns: List of column names
        """
```

---

### tammi.io.json_io

#### class JSONReader

```python
class JSONReader(InputReader):
    def __init__(
        self,
        path: str,
        text_column: str = "text_content",
        id_column: str = "text_id",
        lowercase: bool = True
    ):
        """
        Read from a JSON file.
        
        Supports:
        - Array format: [{"id": ..., "text": ...}, ...]
        - Object format: {"key": [{"id": ..., "text": ...}, ...]}
        """
```

#### class JSONWriter

```python
class JSONWriter(OutputWriter):
    def __init__(self, path: str, columns: List[str]):
        """
        Write to a JSON file.
        
        Output format:
        {
            "metadata": {"columns": [...], "count": N},
            "records": [...]
        }
        """
```

#### class JSONLReader / JSONLWriter

```python
class JSONLReader(InputReader):
    """Read JSON Lines format (one JSON object per line)."""

class JSONLWriter(OutputWriter):
    """Write JSON Lines format (one JSON object per line)."""
```

---

### tammi.io.file_io

#### class TextFileReader

```python
class TextFileReader(InputReader):
    def __init__(
        self,
        paths: List[str],
        extensions: str = ".txt",
        recursive: bool = False,
        lowercase: bool = True
    ):
        """
        Read text files from directories.
        
        Args:
            paths: List of file or directory paths
            extensions: Comma-separated file extensions
            recursive: Whether to recurse into subdirectories
            lowercase: Whether to lowercase text
        """
```

---

### tammi.io.database_io

#### class DatabaseConfig

```python
@dataclass
class DatabaseConfig:
    db_type: str          # 'sqlite', 'mysql', 'postgresql', 'mongodb'
    database: str         # Database name or path
    table: str            # Table/collection name
    text_column: str      # Column containing text
    id_column: str        # Column containing IDs
    host: str = "localhost"
    port: int = 0         # 0 = use default
    username: str = ""
    password: str = ""
    
    @staticmethod
    def _default_port(db_type: str) -> int:
        """Return default port for database type."""
```

#### Database Readers/Writers

```python
class SQLiteReader(InputReader): ...
class SQLiteWriter(OutputWriter): ...
class MySQLReader(InputReader): ...
class MySQLWriter(OutputWriter): ...
class PostgreSQLReader(InputReader): ...
class PostgreSQLWriter(OutputWriter): ...
class MongoDBReader(InputReader): ...
class MongoDBWriter(OutputWriter): ...
```

All database classes take a `DatabaseConfig` in their constructor.

---

## Module: tammi.cli

### tammi.cli.runner

#### class TAMMIRunner

```python
class TAMMIRunner:
    def __init__(
        self,
        morpholex_path: str,
        spacy_model: str = "en_core_web_sm",
        use_gpu: bool = False,
        batch_size: int = 1000,
        n_process: int = 1,
        lowercase: bool = True
    ):
        """
        Initialize the TAMMI runner.
        
        The runner orchestrates the analysis pipeline, coordinating
        readers, the analyzer, and writers.
        """
    
    def run(
        self, 
        reader: InputReader, 
        writer: OutputWriter
    ) -> int:
        """
        Run analysis pipeline.
        
        Args:
            reader: Input reader to get texts from
            writer: Output writer to write results to
            
        Returns:
            Number of texts processed
        """
    
    def run_with_batched_output(
        self,
        reader: InputReader,
        writer: OutputWriter,
        batch_size: int = 100
    ) -> int:
        """
        Run analysis with batched output (for databases).
        
        Writes results in batches rather than all at once,
        which is more efficient for database destinations.
        """
```

#### Helper Functions

```python
def create_reader_from_args(args: argparse.Namespace) -> InputReader:
    """Create an InputReader based on CLI arguments."""

def create_writer_from_args(args: argparse.Namespace) -> OutputWriter:
    """Create an OutputWriter based on CLI arguments."""
```

---

### tammi.cli.progress

#### class ProgressBar

```python
class ProgressBar:
    def __init__(self, total: int, width: int = 30):
        """
        Create a progress bar.
        
        Args:
            total: Total number of items
            width: Width of the bar in characters
        """
    
    def update(self, current: int) -> None:
        """Update progress bar to current position."""
    
    def finish(self) -> None:
        """Complete the progress bar."""
```

#### Functions

```python
def quick_cpu_probe() -> Tuple[int, int, Optional[float]]:
    """
    Quick benchmark to suggest optimal settings.
    
    Returns:
        (suggested_n_process, suggested_batch_size, tokens_per_sec)
    """

def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU is available for spaCy.
    
    Returns:
        (is_available, message)
    """
```

---

### tammi.cli.menu

```python
def interactive_menu(
    args: argparse.Namespace,
    suggested_n_process: int,
    suggested_batch: int
) -> argparse.Namespace:
    """
    Launch interactive menu for configuration.
    
    Args:
        args: Current argument namespace
        suggested_n_process: Suggested number of processes
        suggested_batch: Suggested batch size
        
    Returns:
        Updated argument namespace
    """
```

---

### tammi.cli.main

```python
def main() -> None:
    """Main CLI entry point."""

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

def run_tammi(args: argparse.Namespace) -> None:
    """Run TAMMI analysis with parsed arguments."""
```
