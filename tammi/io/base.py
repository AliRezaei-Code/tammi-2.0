"""
Base classes and interfaces for I/O operations.

Designed and implemented by Ali Rezaei.

Implements Strategy Pattern for different I/O formats and
Factory Pattern for creating appropriate readers/writers.
"""
from __future__ import annotations

__author__ = "Ali Rezaei"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from tammi.io.database_io import DatabaseConfig


@dataclass
class TextRecord:
    """A single text record with ID and content."""
    text_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ResultRecord:
    """A single result record with metrics."""
    text_id: str
    metrics: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputReader(ABC):
    """
    Abstract base class for input readers (Strategy Pattern).
    
    All input sources must implement this interface to provide
    a consistent way to stream text records.
    """
    
    @abstractmethod
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """
        Stream text records from the source.
        
        Yields:
            Tuple of (text_content, metadata_dict) where metadata contains at least 'text_id'
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the total number of records, or 0 if unknown."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up any resources."""
        pass
    
    def __enter__(self) -> "InputReader":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class OutputWriter(ABC):
    """
    Abstract base class for output writers (Strategy Pattern).
    
    All output destinations must implement this interface to provide
    a consistent way to write result records.
    """
    
    @abstractmethod
    def write_header(self, columns: List[str]) -> None:
        """Write the header/schema for results."""
        pass
    
    @abstractmethod
    def write_record(self, text_id: str, values: List[float]) -> None:
        """Write a single result record."""
        pass
    
    @abstractmethod
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        """
        Write a batch of records.
        
        Returns:
            Number of records written
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Flush and clean up resources."""
        pass
    
    def __enter__(self) -> "OutputWriter":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ReaderFactory:
    """
    Factory for creating appropriate InputReader instances.
    
    Implements Factory Pattern to decouple reader creation from usage.
    """
    
    _readers: Dict[str, Type[InputReader]] = {}
    
    @classmethod
    def register(cls, source_type: str, reader_class: Type[InputReader]) -> None:
        """Register a reader class for a source type."""
        cls._readers[source_type.lower()] = reader_class
    
    @classmethod
    def create(
        cls,
        source_type: str,
        **kwargs: Any,
    ) -> InputReader:
        """
        Create an InputReader for the given source type.
        
        Args:
            source_type: Type of source ('files', 'csv', 'json', 'sqlite', 'mysql', 'postgresql', 'mongodb')
            **kwargs: Arguments specific to the reader type
            
        Returns:
            An InputReader instance
            
        Raises:
            ValueError: If source_type is not registered
        """
        reader_class = cls._readers.get(source_type.lower())
        if reader_class is None:
            available = ", ".join(cls._readers.keys())
            raise ValueError(
                f"Unknown source type: {source_type}. Available: {available}"
            )
        return reader_class(**kwargs)
    
    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of registered source types."""
        return list(cls._readers.keys())


class WriterFactory:
    """
    Factory for creating appropriate OutputWriter instances.
    
    Implements Factory Pattern to decouple writer creation from usage.
    """
    
    _writers: Dict[str, Type[OutputWriter]] = {}
    
    @classmethod
    def register(cls, dest_type: str, writer_class: Type[OutputWriter]) -> None:
        """Register a writer class for a destination type."""
        cls._writers[dest_type.lower()] = writer_class
    
    @classmethod
    def create(
        cls,
        dest_type: str,
        columns: List[str],
        **kwargs: Any,
    ) -> OutputWriter:
        """
        Create an OutputWriter for the given destination type.
        
        Args:
            dest_type: Type of destination ('csv', 'json', 'sqlite', 'mysql', 'postgresql', 'mongodb')
            columns: Column names for the output
            **kwargs: Arguments specific to the writer type
            
        Returns:
            An OutputWriter instance
            
        Raises:
            ValueError: If dest_type is not registered
        """
        writer_class = cls._writers.get(dest_type.lower())
        if writer_class is None:
            available = ", ".join(cls._writers.keys())
            raise ValueError(
                f"Unknown destination type: {dest_type}. Available: {available}"
            )
        # Type ignore: concrete writer classes accept columns parameter
        writer = writer_class(columns=columns, **kwargs)  # type: ignore[call-arg]
        return writer
    
    @classmethod
    def available_types(cls) -> List[str]:
        """Return list of registered destination types."""
        return list(cls._writers.keys())


def check_available_drivers() -> Dict[str, bool]:
    """Check which database/format drivers are available."""
    drivers: Dict[str, bool] = {}
    
    # SQLite (always available in Python)
    try:
        import sqlite3  # noqa: F401
        drivers["sqlite"] = True
    except ImportError:
        drivers["sqlite"] = False
    
    # MySQL
    try:
        import mysql.connector  # noqa: F401  # type: ignore[import-not-found]
        drivers["mysql"] = True
    except ImportError:
        drivers["mysql"] = False
    
    # PostgreSQL
    try:
        import psycopg2  # noqa: F401  # type: ignore[import-not-found]
        drivers["postgresql"] = True
    except ImportError:
        drivers["postgresql"] = False
    
    # MongoDB
    try:
        import pymongo  # noqa: F401  # type: ignore[import-not-found]
        drivers["mongodb"] = True
    except ImportError:
        drivers["mongodb"] = False
    
    # JSON (always available)
    drivers["json"] = True
    
    # CSV (always available)
    drivers["csv"] = True
    
    return drivers


# Global driver availability cache
AVAILABLE_DRIVERS = check_available_drivers()
