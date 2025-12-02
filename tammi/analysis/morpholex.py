"""MorphoLex dictionary loader and utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class MorphoLexDict:
    """
    Loader and container for the MorphoLex dictionary.
    
    The MorphoLex dictionary contains morphological information for English words,
    including prefix counts, suffix counts, root counts, and various frequency metrics.
    """
    
    def __init__(self, path: Optional[Path | str] = None) -> None:
        self._data: Dict[str, List[Any]] = {}
        self._path: Optional[Path] = Path(path) if path else None
        
        if self._path:
            self.load(self._path)
    
    def load(self, path: Path | str) -> None:
        """
        Load MorphoLex data from a CSV file.
        
        The CSV is expected to have no header, with the first column being
        the word and columns 2-76 containing morphological metrics.
        """
        path = Path(path)
        self._path = path
        self._data.clear()
        
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                key = row[0]
                raw_values = row[1:76]
                values: List[Any] = []
                for val in raw_values:
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(val)
                self._data[key] = values
    
    def get(self, word: str) -> Optional[List[Any]]:
        """Get morphological data for a word."""
        return self._data.get(word)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in dictionary."""
        return word in self._data
    
    def __len__(self) -> int:
        """Return number of words in dictionary."""
        return len(self._data)
    
    def __getitem__(self, word: str) -> List[Any]:
        """Get morphological data for a word (raises KeyError if not found)."""
        return self._data[word]
    
    @property
    def path(self) -> Optional[Path]:
        """Return the path to the loaded dictionary file."""
        return self._path
    
    @property
    def data(self) -> Dict[str, List[Any]]:
        """Return the raw dictionary data."""
        return self._data


def discover_morpholex_files(base_path: Path = Path(".")) -> List[Path]:
    """Find potential MorphoLex CSV files in a directory."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            if "morph" in item.name.lower() or "lex" in item.name.lower():
                candidates.append(item)
    return sorted(candidates)
