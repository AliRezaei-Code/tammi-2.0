"""
I/O module for TAMMI - handles reading and writing data from various sources.

Designed and implemented by Ali Rezaei using:
- Strategy Pattern: InputReader/OutputWriter abstract base classes
- Factory Pattern: ReaderFactory/WriterFactory for creating readers/writers

Supported formats:
- Text files (directories of .txt files)
- CSV files
- JSON and JSONL files
- Databases: SQLite, MySQL, PostgreSQL, MongoDB
"""

__author__ = "Ali Rezaei"

from tammi.io.base import (
    InputReader,
    OutputWriter,
    ReaderFactory,
    WriterFactory,
)
from tammi.io.csv_io import CSVReader, CSVWriter
from tammi.io.json_io import JSONReader, JSONWriter
from tammi.io.file_io import TextFileReader
from tammi.io.database_io import (
    DatabaseConfig,
    SQLiteReader,
    SQLiteWriter,
    MySQLReader,
    MySQLWriter,
    PostgreSQLReader,
    PostgreSQLWriter,
    MongoDBReader,
    MongoDBWriter,
)

__all__ = [
    "InputReader",
    "OutputWriter",
    "ReaderFactory",
    "WriterFactory",
    "CSVReader",
    "CSVWriter",
    "JSONReader",
    "JSONWriter",
    "TextFileReader",
    "DatabaseConfig",
    "SQLiteReader",
    "SQLiteWriter",
    "MySQLReader",
    "MySQLWriter",
    "PostgreSQLReader",
    "PostgreSQLWriter",
    "MongoDBReader",
    "MongoDBWriter",
]
