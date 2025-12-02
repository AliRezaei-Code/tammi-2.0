"""CSV input/output handlers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from tammi.io.base import InputReader, OutputWriter, ReaderFactory, WriterFactory


class CSVReader(InputReader):
    """Read text records from a CSV file."""
    
    def __init__(
        self,
        path: str | Path,
        text_column: str = "text_content",
        id_column: str = "text_id",
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        self.path = Path(path)
        self.text_column = text_column
        self.id_column = id_column
        self.encoding = encoding
        self._file = None
        self._reader = None
        self._count: int | None = None
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from CSV."""
        with self.path.open(newline="", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(self.text_column, "")
                text_id = row.get(self.id_column, "")
                if lowercase:
                    text = text.lower()
                yield text, {"text_id": text_id}
    
    def count(self) -> int:
        """Count total rows in CSV."""
        if self._count is None:
            with self.path.open(newline="", encoding=self.encoding) as f:
                self._count = sum(1 for _ in f) - 1  # subtract header
        return self._count
    
    def close(self) -> None:
        """Nothing to close for CSV reader."""
        pass


class CSVWriter(OutputWriter):
    """Write result records to a CSV file."""
    
    def __init__(
        self,
        path: str | Path,
        columns: List[str],
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        self.path = Path(path)
        self.columns = columns
        self.encoding = encoding
        self._file = self.path.open("w", newline="", encoding=encoding)
        self._writer = csv.writer(self._file)
        self._header_written = False
    
    def write_header(self, columns: List[str]) -> None:
        """Write CSV header row."""
        if not self._header_written:
            self._writer.writerow(["text_id", *columns])
            self._header_written = True
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        """Write a single result row."""
        if not self._header_written:
            self.write_header(self.columns)
        self._writer.writerow([text_id, *values])
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        """Write multiple records."""
        if not self._header_written:
            self.write_header(self.columns)
        for text_id, values in records:
            self._writer.writerow([text_id, *values])
        return len(records)
    
    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None


def get_csv_columns(csv_path: Path) -> List[str]:
    """Get column names from a CSV file."""
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


# Register with factories
ReaderFactory.register("csv", CSVReader)
WriterFactory.register("csv", CSVWriter)
