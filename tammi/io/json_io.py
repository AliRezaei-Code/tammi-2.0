"""JSON input/output handlers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

from tammi.io.base import InputReader, OutputWriter, ReaderFactory, WriterFactory


class JSONReader(InputReader):
    """
    Read text records from a JSON file.
    
    Supports two formats:
    1. Array of objects: [{"id": "1", "text": "..."}, ...]
    2. Object with records key: {"records": [...]}
    """
    
    def __init__(
        self,
        path: str | Path,
        text_column: str = "text_content",
        id_column: str = "text_id",
        records_key: str | None = None,  # If data is nested under a key
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        self.path = Path(path)
        self.text_column = text_column
        self.id_column = id_column
        self.records_key = records_key
        self.encoding = encoding
        self._data: List[Dict[str, Any]] | None = None
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and parse JSON data."""
        if self._data is not None:
            return self._data
            
        with self.path.open(encoding=self.encoding) as f:
            raw_data = json.load(f)
        
        data: List[Dict[str, Any]]
        
        # Handle different JSON structures
        if isinstance(raw_data, list):
            data = raw_data
        elif isinstance(raw_data, dict):
            if self.records_key and self.records_key in raw_data:
                data = raw_data[self.records_key]
            elif "records" in raw_data:
                data = raw_data["records"]
            elif "data" in raw_data:
                data = raw_data["data"]
            else:
                # Try to find a list value
                found_data: List[Dict[str, Any]] | None = None
                for value in raw_data.values():
                    if isinstance(value, list):
                        found_data = value
                        break
                if found_data is None:
                    raise ValueError(
                        f"Could not find records in JSON. "
                        f"Specify records_key or use array format."
                    )
                data = found_data
        else:
            raise ValueError(f"Unexpected JSON format: {type(raw_data)}")
        
        self._data = data
        return data
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from JSON."""
        data = self._load_data()
        for record in data:
            text = str(record.get(self.text_column, ""))
            text_id = str(record.get(self.id_column, ""))
            if lowercase:
                text = text.lower()
            yield text, {"text_id": text_id}
    
    def count(self) -> int:
        """Count total records in JSON."""
        return len(self._load_data())
    
    def close(self) -> None:
        """Clear cached data."""
        self._data = None


class JSONWriter(OutputWriter):
    """
    Write result records to a JSON file.
    
    Output format:
    {
        "metadata": {"columns": [...], "count": N},
        "records": [{"text_id": "...", "metric1": 0.5, ...}, ...]
    }
    """
    
    def __init__(
        self,
        path: str | Path,
        columns: List[str],
        encoding: str = "utf-8",
        indent: int = 2,
        **kwargs: Any,
    ) -> None:
        self.path = Path(path)
        self.columns = columns
        self.encoding = encoding
        self.indent = indent
        self._records: List[Dict[str, Union[str, float]]] = []
        self._header_written = False
    
    def write_header(self, columns: List[str]) -> None:
        """Store columns for later use (JSON writes everything at close)."""
        self.columns = columns
        self._header_written = True
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        """Buffer a single result record."""
        if not self._header_written:
            self.write_header(self.columns)
        
        record: Dict[str, Union[str, float]] = {"text_id": text_id}
        for col, val in zip(self.columns, values):
            record[col] = val
        self._records.append(record)
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        """Buffer multiple records."""
        if not self._header_written:
            self.write_header(self.columns)
        
        for text_id, values in records:
            record: Dict[str, Union[str, float]] = {"text_id": text_id}
            for col, val in zip(self.columns, values):
                record[col] = val
            self._records.append(record)
        return len(records)
    
    def close(self) -> None:
        """Write all buffered records to file."""
        output = {
            "metadata": {
                "columns": ["text_id", *self.columns],
                "count": len(self._records),
            },
            "records": self._records,
        }
        
        with self.path.open("w", encoding=self.encoding) as f:
            json.dump(output, f, indent=self.indent)
        
        self._records = []


class JSONLReader(InputReader):
    """
    Read text records from a JSON Lines file (one JSON object per line).
    """
    
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
        self._count: int | None = None
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from JSON Lines file."""
        with self.path.open(encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = str(record.get(self.text_column, ""))
                text_id = str(record.get(self.id_column, ""))
                if lowercase:
                    text = text.lower()
                yield text, {"text_id": text_id}
    
    def count(self) -> int:
        """Count total records."""
        if self._count is None:
            with self.path.open(encoding=self.encoding) as f:
                self._count = sum(1 for line in f if line.strip())
        return self._count
    
    def close(self) -> None:
        """Nothing to close."""
        pass


class JSONLWriter(OutputWriter):
    """
    Write result records to a JSON Lines file (one JSON object per line).
    More memory-efficient for large datasets.
    """
    
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
        self._file: Any = self.path.open("w", encoding=encoding)
        self._header_written = False
    
    def write_header(self, columns: List[str]) -> None:
        """Store columns (no header line in JSONL)."""
        self.columns = columns
        self._header_written = True
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        """Write a single record as a JSON line."""
        if not self._header_written:
            self.write_header(self.columns)
        
        record: Dict[str, Union[str, float]] = {"text_id": text_id}
        for col, val in zip(self.columns, values):
            record[col] = val
        
        if self._file:
            self._file.write(json.dumps(record) + "\n")
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        """Write multiple records."""
        for text_id, values in records:
            self.write_record(text_id, values)
        return len(records)
    
    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None


# Register with factories
ReaderFactory.register("json", JSONReader)
WriterFactory.register("json", JSONWriter)
ReaderFactory.register("jsonl", JSONLReader)
WriterFactory.register("jsonl", JSONLWriter)
