"""Text file input handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from tammi.io.base import InputReader, ReaderFactory


class TextFileReader(InputReader):
    """
    Read text content from text files in a directory.
    """
    
    def __init__(
        self,
        paths: List[str] | List[Path],
        extensions: Tuple[str, ...] = (".txt",),
        recursive: bool = False,
        encoding: str = "utf-8",
        fallback_encoding: str = "latin-1",
        **kwargs: Any,
    ) -> None:
        self.input_paths = [Path(p) for p in paths]
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.recursive = recursive
        self.encoding = encoding
        self.fallback_encoding = fallback_encoding
        self._file_paths: List[Path] | None = None
    
    def _discover_files(self) -> List[Path]:
        """Find all matching text files."""
        if self._file_paths is not None:
            return self._file_paths
        
        self._file_paths = []
        for input_path in self.input_paths:
            if input_path.is_file():
                if input_path.suffix.lower() in self.extensions:
                    self._file_paths.append(input_path)
            elif input_path.is_dir():
                iterator = input_path.rglob("*") if self.recursive else input_path.glob("*")
                for candidate in iterator:
                    if candidate.is_file() and candidate.suffix.lower() in self.extensions:
                        self._file_paths.append(candidate)
        
        return self._file_paths
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text content from files."""
        for path in self._discover_files():
            try:
                text = path.read_text(encoding=self.encoding)
            except UnicodeDecodeError:
                text = path.read_text(encoding=self.fallback_encoding)
            
            if lowercase:
                text = text.lower()
            
            yield text, {"text_id": str(path)}
    
    def count(self) -> int:
        """Count total number of files."""
        return len(self._discover_files())
    
    def close(self) -> None:
        """Nothing to close."""
        pass


def discover_text_folders(base_path: Path = Path(".")) -> List[Path]:
    """Find directories that contain text files."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith((".", "_")):
            # Check if it has any .txt files
            txt_files = list(item.glob("*.txt"))[:5]
            if txt_files:
                candidates.append(item)
    return sorted(candidates)


# Register with factory
ReaderFactory.register("files", TextFileReader)
