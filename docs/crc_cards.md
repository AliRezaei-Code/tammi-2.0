# TAMMI CRC Cards

## Class-Responsibility-Collaboration Diagrams

CRC (Class-Responsibility-Collaboration) cards describe the responsibilities of each class and which other classes they collaborate with.

---

## Core Analysis Classes

### TAMMIAnalyzer

| **Class Name** | TAMMIAnalyzer |
|----------------|---------------|
| **Module** | `tammi.analysis.analyzer` |
| **Superclass** | None |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Load and initialize spaCy model | spaCy |
| Load MorphoLex dictionary | MorphoLexDict |
| Analyze single text for morphological features | MorphoLexDict |
| Calculate inflectional metrics | metrics module |
| Calculate derivational metrics | metrics module |
| Calculate MCI (Morphological Complexity Index) | metrics module |
| Calculate TTR (Type-Token Ratio) | metrics module |
| Return analysis results as dictionary | - |

---

### MorphoLexDict

| **Class Name** | MorphoLexDict |
|----------------|---------------|
| **Module** | `tammi.analysis.morpholex` |
| **Superclass** | None |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Load MorphoLex CSV file | csv (stdlib) |
| Store morpheme data in memory | - |
| Provide fast lookup by word | - |
| Return morpheme features for words | - |
| Handle missing words gracefully | - |

---

## I/O Base Classes (Strategy Pattern)

### InputReader (Abstract)

| **Class Name** | InputReader |
|----------------|-------------|
| **Module** | `tammi.io.base` |
| **Superclass** | ABC |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Define interface for reading text data | - |
| Stream (text_id, text_content) tuples | - |
| Count total number of items | - |
| Clean up resources on close | - |

---

### OutputWriter (Abstract)

| **Class Name** | OutputWriter |
|----------------|--------------|
| **Module** | `tammi.io.base` |
| **Superclass** | ABC |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Define interface for writing results | - |
| Write single result row | - |
| Write batch of results | - |
| Write all results at once | - |
| Clean up resources on close | - |

---

### ReaderFactory

| **Class Name** | ReaderFactory |
|----------------|---------------|
| **Module** | `tammi.io.base` |
| **Superclass** | None |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Register reader classes by type name | InputReader subclasses |
| Create reader instances by type | InputReader subclasses |
| Report available reader types | - |

---

### WriterFactory

| **Class Name** | WriterFactory |
|----------------|---------------|
| **Module** | `tammi.io.base` |
| **Superclass** | None |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Register writer classes by type name | OutputWriter subclasses |
| Create writer instances by type | OutputWriter subclasses |
| Report available writer types | - |

---

## Concrete I/O Classes

### CSVReader

| **Class Name** | CSVReader |
|----------------|-----------|
| **Module** | `tammi.io.csv_io` |
| **Superclass** | InputReader |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Read CSV files with headers | csv (stdlib) |
| Extract text and ID columns | - |
| Optionally lowercase text | - |
| Stream rows as tuples | - |

---

### CSVWriter

| **Class Name** | CSVWriter |
|----------------|-----------|
| **Module** | `tammi.io.csv_io` |
| **Superclass** | OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Write CSV files with headers | csv (stdlib) |
| Write single or batch rows | - |
| Handle file I/O | - |

---

### JSONReader

| **Class Name** | JSONReader |
|----------------|------------|
| **Module** | `tammi.io.json_io` |
| **Superclass** | InputReader |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Read JSON files (array or object format) | json (stdlib) |
| Auto-detect JSON structure | - |
| Extract text and ID fields | - |
| Stream records as tuples | - |

---

### JSONWriter

| **Class Name** | JSONWriter |
|----------------|------------|
| **Module** | `tammi.io.json_io` |
| **Superclass** | OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Write JSON files with metadata | json (stdlib) |
| Collect results in memory | - |
| Output formatted JSON | - |

---

### JSONLReader

| **Class Name** | JSONLReader |
|----------------|-------------|
| **Module** | `tammi.io.json_io` |
| **Superclass** | InputReader |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Read JSON Lines files | json (stdlib) |
| Stream one record per line | - |
| Handle large files efficiently | - |

---

### JSONLWriter

| **Class Name** | JSONLWriter |
|----------------|-------------|
| **Module** | `tammi.io.json_io` |
| **Superclass** | OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Write JSON Lines files | json (stdlib) |
| Write one JSON object per line | - |
| Stream-friendly output | - |

---

### TextFileReader

| **Class Name** | TextFileReader |
|----------------|----------------|
| **Module** | `tammi.io.file_io` |
| **Superclass** | InputReader |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Read text files from directories | pathlib (stdlib) |
| Filter by file extension | - |
| Optionally recurse subdirectories | - |
| Use filename as text ID | - |

---

### SQLiteReader

| **Class Name** | SQLiteReader |
|----------------|--------------|
| **Module** | `tammi.io.database_io` |
| **Superclass** | InputReader |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Connect to SQLite database | sqlite3 (stdlib) |
| Query text and ID columns | DatabaseConfig |
| Stream database rows | - |

---

### SQLiteWriter

| **Class Name** | SQLiteWriter |
|----------------|--------------|
| **Module** | `tammi.io.database_io` |
| **Superclass** | OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Connect to SQLite database | sqlite3 (stdlib) |
| Create results table if needed | DatabaseConfig |
| Insert result rows | - |
| Handle transactions | - |

---

### MySQLReader / MySQLWriter

| **Class Name** | MySQLReader, MySQLWriter |
|----------------|--------------------------|
| **Module** | `tammi.io.database_io` |
| **Superclass** | InputReader, OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Connect to MySQL database | mysql-connector-python |
| Query/write text data | DatabaseConfig |
| Handle MySQL-specific syntax | - |

---

### PostgreSQLReader / PostgreSQLWriter

| **Class Name** | PostgreSQLReader, PostgreSQLWriter |
|----------------|-------------------------------------|
| **Module** | `tammi.io.database_io` |
| **Superclass** | InputReader, OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Connect to PostgreSQL database | psycopg2 |
| Query/write text data | DatabaseConfig |
| Handle PostgreSQL-specific syntax | - |

---

### MongoDBReader / MongoDBWriter

| **Class Name** | MongoDBReader, MongoDBWriter |
|----------------|-------------------------------|
| **Module** | `tammi.io.database_io` |
| **Superclass** | InputReader, OutputWriter |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Connect to MongoDB | pymongo |
| Query/write documents | DatabaseConfig |
| Handle document-based data model | - |

---

### DatabaseConfig

| **Class Name** | DatabaseConfig |
|----------------|----------------|
| **Module** | `tammi.io.database_io` |
| **Superclass** | None (dataclass) |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Store database connection parameters | - |
| Provide default ports per database type | - |
| Validate configuration | - |

---

## CLI Classes

### TAMMIRunner

| **Class Name** | TAMMIRunner |
|----------------|-------------|
| **Module** | `tammi.cli.runner` |
| **Superclass** | None |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Orchestrate analysis pipeline | TAMMIAnalyzer |
| Coordinate InputReader and OutputWriter | InputReader, OutputWriter |
| Display progress during processing | ProgressBar |
| Handle batched output for databases | - |
| Return processing statistics | - |

---

### ProgressBar

| **Class Name** | ProgressBar |
|----------------|-------------|
| **Module** | `tammi.cli.progress` |
| **Superclass** | None |

| **Responsibilities** | **Collaborators** |
|---------------------|-------------------|
| Display progress bar in terminal | - |
| Calculate and show ETA | - |
| Handle different terminal widths | - |
| Update display smoothly | - |

---

## Collaboration Diagram

```
                              ┌─────────────────┐
                              │   CLI (main.py) │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  TAMMIRunner    │
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  InputReader    │      │  TAMMIAnalyzer  │      │  OutputWriter   │
    │  (Strategy)     │      │                 │      │  (Strategy)     │
    └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
             │                        │                        │
    ┌────────┴────────┐      ┌────────┴────────┐      ┌────────┴────────┐
    │                 │      │                 │      │                 │
    ▼                 ▼      ▼                 ▼      ▼                 ▼
┌────────┐      ┌────────┐  ┌────────┐  ┌───────────┐ ┌────────┐ ┌────────┐
│CSVReader│     │JSONReader│ │MorphoLex│ │  spaCy   │ │CSVWriter│ │JSONWriter│
└────────┘      └────────┘  │  Dict   │ │          │ └────────┘ └────────┘
                            └────────┘  └──────────┘
```

---

## Sequence Diagram: Typical Analysis Run

```
User          CLI          TAMMIRunner    InputReader    TAMMIAnalyzer    OutputWriter
  │            │               │              │               │               │
  │──command──▶│               │              │               │               │
  │            │──create()────▶│              │               │               │
  │            │               │──create()───▶│               │               │
  │            │               │              │◀──────────────│               │
  │            │               │──create()───────────────────▶│               │
  │            │               │              │               │◀──────────────│
  │            │               │──create()───────────────────────────────────▶│
  │            │               │              │               │               │◀─
  │            │──run()───────▶│              │               │               │
  │            │               │              │               │               │
  │            │               │ loop:        │               │               │
  │            │               │──stream()───▶│               │               │
  │            │               │◀─(id,text)───│               │               │
  │            │               │──analyze()──────────────────▶│               │
  │            │               │◀──results────────────────────│               │
  │            │               │──write()────────────────────────────────────▶│
  │            │               │              │               │               │
  │            │               │──close()────▶│               │               │
  │            │               │──close()────────────────────────────────────▶│
  │            │◀──count───────│              │               │               │
  │◀──done─────│               │              │               │               │
```
