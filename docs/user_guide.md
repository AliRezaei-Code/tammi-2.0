# TAMMI User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Usage](#cli-usage)
4. [Python API](#python-api)
5. [Input Formats](#input-formats)
6. [Output Formats](#output-formats)
7. [Database Support](#database-support)
8. [GPU Acceleration](#gpu-acceleration)
9. [Understanding the Output](#understanding-the-output)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tammi.git
cd tammi

# Install the package
pip install -e .

# Download required spaCy model
python -m spacy download en_core_web_sm
```

### Installation with Optional Dependencies

```bash
# For MySQL support
pip install -e ".[mysql]"

# For PostgreSQL support
pip install -e ".[postgresql]"

# For MongoDB support
pip install -e ".[mongodb]"

# For GPU support
pip install -e ".[gpu]"

# For all optional dependencies
pip install -e ".[all]"

# For development (testing, linting)
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check CLI is working
python tammi_cli.py --help

# Or if installed as a package
tammi --help
```

---

## Quick Start

### Analyze Text Files

```bash
# Analyze all .txt files in a directory
python tammi_cli.py my_texts/ -o results.csv

# Analyze specific files
python tammi_cli.py file1.txt file2.txt -o results.csv
```

### Analyze CSV Data

```bash
# Analyze texts from a CSV file
python tammi_cli.py --input-csv data.csv --text-column content -o results.csv
```

### Analyze JSON Data

```bash
# Analyze texts from a JSON file
python tammi_cli.py --input-json data.json --text-column text -o results.json --output-json
```

### Interactive Mode

```bash
# Launch interactive menu
python tammi_cli.py --menu
```

---

## CLI Usage

### Basic Syntax

```bash
python tammi_cli.py [INPUT_OPTIONS] [OUTPUT_OPTIONS] [PROCESSING_OPTIONS]
```

### Input Options

| Option | Description | Example |
|--------|-------------|---------|
| `inputs` | Text files or directories | `texts/` or `file.txt` |
| `--input-csv PATH` | Read from CSV file | `--input-csv data.csv` |
| `--input-json PATH` | Read from JSON/JSONL file | `--input-json data.json` |
| `--input-db CONN` | Read from database | `--input-db sqlite:data.db` |
| `--text-column NAME` | Column/field containing text | `--text-column content` |
| `--id-column NAME` | Column/field containing ID | `--id-column doc_id` |
| `--ext EXT` | File extensions to include | `--ext .txt,.md` |
| `--recursive` | Recurse into subdirectories | `--recursive` |

### Output Options

| Option | Description | Example |
|--------|-------------|---------|
| `-o, --output PATH` | Output file path | `-o results.csv` |
| `--output-json` | Output as JSON format | `--output-json` |
| `--output-db CONN` | Write to database | `--output-db sqlite:results.db` |
| `--output-table NAME` | Table name for DB output | `--output-table tammi_results` |

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model NAME` | spaCy model name | `en_core_web_sm` |
| `--morpholex PATH` | MorphoLex CSV path | `morpho_lex_df_w_log_w_prefsuf_no_head.csv` |
| `--batch-size N` | spaCy pipe batch size | Auto-detected |
| `--n-process N` | Number of processes | Auto-detected |
| `--keep-case` | Don't lowercase text | Off (lowercase by default) |
| `--use-gpu` | Enable GPU acceleration | Off |
| `--menu` | Launch interactive menu | Off |

### Example Commands

```bash
# Basic: Process text files
python tammi_cli.py responses/ -o morphemes.csv

# With options: Recursive, multiple extensions, custom model
python tammi_cli.py texts/ --recursive --ext .txt,.md -m en_core_web_lg -o results.csv

# CSV input with custom columns
python tammi_cli.py --input-csv essays.csv --text-column essay_text --id-column student_id -o essay_analysis.csv

# JSON input to JSON output
python tammi_cli.py --input-json documents.json --text-column body -o analysis.json --output-json

# Database to database
python tammi_cli.py --input-db sqlite:texts.db --input-table documents --output-db sqlite:results.db

# Using GPU with large batches
python tammi_cli.py large_corpus/ --use-gpu --batch-size 5000 -o results.csv
```

---

## Python API

### Basic Usage

```python
from tammi.analysis.analyzer import TAMMIAnalyzer

# Initialize analyzer
analyzer = TAMMIAnalyzer(
    morpholex_path="morpho_lex_df_w_log_w_prefsuf_no_head.csv",
    spacy_model="en_core_web_sm",
    use_gpu=False
)

# Analyze a single text
text = "The quickly running dogs were happily playing in the beautiful garden."
results = analyzer.analyze_text(text, text_id="sample1")

print(results)
# {'text_id': 'sample1', 'Inflected_Tokens': 0.42, 'Derivational_Tokens': 0.28, ...}
```

### Using Readers and Writers

```python
from tammi.io.csv_io import CSVReader, CSVWriter
from tammi.io.json_io import JSONReader, JSONWriter
from tammi.analysis.analyzer import TAMMIAnalyzer
from tammi.analysis.metrics import COLUMN_NAMES

# Read from CSV
reader = CSVReader(
    path="data.csv",
    text_column="content",
    id_column="doc_id",
    lowercase=True
)

# Initialize analyzer
analyzer = TAMMIAnalyzer("morpho_lex_df_w_log_w_prefsuf_no_head.csv")

# Process and collect results
results = []
for text_id, text in reader.stream():
    result = analyzer.analyze_text(text, text_id)
    results.append(result)

reader.close()

# Write to JSON
writer = JSONWriter(path="results.json", columns=COLUMN_NAMES)
writer.write_all(results)
writer.close()
```

### Using the Runner

```python
from tammi.cli.runner import TAMMIRunner, create_reader_from_args
from tammi.io.csv_io import CSVWriter
from argparse import Namespace

# Create a namespace with arguments
args = Namespace(
    input_source_type="csv",
    input_csv_path="data.csv",
    input_text_column="content",
    input_id_column="doc_id",
    lowercase=True
)

# Create reader
reader = create_reader_from_args(args)

# Create writer
writer = CSVWriter(path="results.csv", columns=COLUMN_NAMES)

# Create and run runner
runner = TAMMIRunner(
    morpholex_path="morpho_lex_df_w_log_w_prefsuf_no_head.csv",
    spacy_model="en_core_web_sm"
)

count = runner.run(reader, writer)
print(f"Processed {count} documents")
```

### Using Factories

```python
from tammi.io.base import ReaderFactory, WriterFactory
from tammi.analysis.metrics import COLUMN_NAMES

# Create reader via factory
reader = ReaderFactory.create(
    "json",
    path="data.json",
    text_column="text",
    id_column="id"
)

# Create writer via factory
writer = WriterFactory.create(
    "csv",
    path="output.csv",
    columns=COLUMN_NAMES
)

# Check available types
print(ReaderFactory.available_types())  # ['csv', 'json', 'jsonl', 'files', 'sqlite', ...]
print(WriterFactory.available_types())  # ['csv', 'json', 'jsonl', 'sqlite', ...]
```

---

## Input Formats

### Text Files

Place `.txt` files in a directory:

```text
texts/
├── document1.txt
├── document2.txt
└── subfolder/
    └── document3.txt
```

```bash
python tammi_cli.py texts/ --recursive -o results.csv
```

### CSV Format

```csv
doc_id,content,author
1,"The cat sat on the mat.",John
2,"Running quickly through the forest.",Jane
```

```bash
python tammi_cli.py --input-csv data.csv --text-column content --id-column doc_id -o results.csv
```

### JSON Format

**Array format:**

```json
[
    {"id": "doc1", "text": "The cat sat on the mat."},
    {"id": "doc2", "text": "Running quickly through the forest."}
]
```

**Object format:**

```json
{
    "documents": [
        {"id": "doc1", "text": "The cat sat on the mat."},
        {"id": "doc2", "text": "Running quickly through the forest."}
    ]
}
```

```bash
python tammi_cli.py --input-json data.json --text-column text --id-column id -o results.csv
```

### JSONL Format (JSON Lines)

```jsonl
{"id": "doc1", "text": "The cat sat on the mat."}
{"id": "doc2", "text": "Running quickly through the forest."}
```

```bash
python tammi_cli.py --input-json data.jsonl --text-column text --id-column id -o results.csv
```

---

## Output Formats

### CSV Output

Default output format with headers:

```csv
text_id,Inflected_Tokens,Derivational_Tokens,...
doc1,0.42,0.28,...
doc2,0.35,0.31,...
```

### JSON Output

```bash
python tammi_cli.py texts/ -o results.json --output-json
```

```json
{
    "metadata": {
        "columns": ["text_id", "Inflected_Tokens", ...],
        "count": 100
    },
    "records": [
        {"text_id": "doc1", "Inflected_Tokens": 0.42, ...},
        {"text_id": "doc2", "Inflected_Tokens": 0.35, ...}
    ]
}
```

---

## Database Support

### SQLite

```bash
# Read from SQLite
python tammi_cli.py --input-db sqlite:texts.db --input-table documents -o results.csv

# Write to SQLite
python tammi_cli.py texts/ --output-db sqlite:results.db --output-table tammi_results
```

### MySQL

Requires: `pip install mysql-connector-python`

```bash
# Connection string format
python tammi_cli.py --input-db mysql://user:password@localhost:3306/mydb --input-table texts -o results.csv
```

### PostgreSQL

Requires: `pip install psycopg2-binary`

```bash
# Connection string format
python tammi_cli.py --input-db postgresql://user:password@localhost:5432/mydb --input-table texts -o results.csv
```

### MongoDB

Requires: `pip install pymongo`

```bash
# Connection string format
python tammi_cli.py --input-db mongodb://user:password@localhost:27017/mydb --input-table texts_collection -o results.csv
```

---

## GPU Acceleration

### CUDA (NVIDIA)

```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run with GPU
python tammi_cli.py texts/ --use-gpu -o results.csv
```

### MPS (Apple Silicon)

```bash
# Install PyTorch (MPS is included)
pip install torch

# Run with GPU
python tammi_cli.py texts/ --use-gpu -o results.csv
```

**Note:** GPU mode automatically sets `n_process=1` as required by spaCy.

---

## Understanding the Output

TAMMI produces 43 morphological metrics for each text:

### Basic Counts (Normalized)

| Metric | Description |
|--------|-------------|
| `Inflected_Tokens` | Proportion of content words with inflections |
| `Derivational_Tokens` | Proportion of content words with derivational morphemes |
| `Tokens_w_Prefixes` | Proportion of content words with prefixes |
| `Tokens_w_Affixes` | Proportion of content words with affixes |
| `Compounds` | Proportion of compound words |

### Morpheme Counts (Normalized)

| Metric | Description |
|--------|-------------|
| `number_prefixes` | Average number of prefixes per content word |
| `number_roots` | Average number of roots per content word |
| `number_suffixes` | Average number of suffixes per content word |
| `number_affixes` | Average number of affixes per content word |

### MorphoLex Metrics

| Prefix | Description |
|--------|-------------|
| `prefix_*` | Metrics related to prefixes |
| `root_*` | Metrics related to roots |
| `suffix_*` | Metrics related to suffixes |
| `affix_*` | Metrics related to all affixes |

| Suffix | Description |
|--------|-------------|
| `*_family_size` | Morphological family size |
| `*_freq` | Raw frequency |
| `*_log_freq` | Log frequency |
| `*_len` | Average length |

### Complexity Indices

| Metric | Description |
|--------|-------------|
| `mean subset inflectional variety (10)` | Within-subset variety for inflections |
| `inflectional TTR (10)` | Type-token ratio for inflections |
| `inflectional MCI (10)` | Morphological Complexity Index (inflections) |
| `mean subset derivational variety (10)` | Within-subset variety for derivations |
| `derivational TTR (10)` | Type-token ratio for derivations |
| `derivational MCI (10)` | Morphological Complexity Index (derivations) |

---

## Troubleshooting

### Common Issues

#### 1. spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

#### 2. MorphoLex file not found

```bash
# Ensure the file is in the working directory or specify path
python tammi_cli.py texts/ --morpholex /path/to/morpho_lex_df_w_log_w_prefsuf_no_head.csv
```

#### 3. MySQL/PostgreSQL/MongoDB not available

```bash
# Install the required driver
pip install mysql-connector-python  # For MySQL
pip install psycopg2-binary         # For PostgreSQL
pip install pymongo                 # For MongoDB
```

#### 4. GPU not detected

```bash
# Check if PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available())"  # CUDA
python -c "import torch; print(torch.backends.mps.is_available())"  # MPS
```

#### 5. Out of memory errors

```bash
# Reduce batch size
python tammi_cli.py texts/ --batch-size 100 -o results.csv
```

### Getting Help

```bash
# Show all options
python tammi_cli.py --help

# Check version
python -c "import tammi; print(tammi.__version__)"
```
