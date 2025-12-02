"""TAMMI runner - orchestrates the analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

from tammi.analysis.analyzer import TAMMIAnalyzer
from tammi.analysis.metrics import COLUMN_NAMES
from tammi.io.base import InputReader, OutputWriter, ReaderFactory, WriterFactory
from tammi.cli.progress import ProgressBar

if TYPE_CHECKING:
    from tammi.io.database_io import DatabaseConfig


class TAMMIRunner:
    """
    Orchestrates TAMMI analysis pipeline.
    
    Uses Strategy pattern via InputReader/OutputWriter interfaces.
    """
    
    def __init__(
        self,
        morpholex_path: str,
        spacy_model: str = "en_core_web_sm",
        use_gpu: bool = False,
        batch_size: int = 1000,
        n_process: int = 1,
        lowercase: bool = True,
    ) -> None:
        self.morpholex_path = morpholex_path
        self.spacy_model = spacy_model
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.n_process = n_process if not use_gpu else 1  # GPU requires single process
        self.lowercase = lowercase
        
        self._analyzer: Optional[TAMMIAnalyzer] = None
    
    @property
    def analyzer(self) -> TAMMIAnalyzer:
        """Lazy-load the analyzer."""
        if self._analyzer is None:
            self._analyzer = TAMMIAnalyzer(
                morpholex=self.morpholex_path,
                spacy_model=self.spacy_model,
                use_gpu=self.use_gpu,
            )
        return self._analyzer
    
    def run(
        self,
        reader: InputReader,
        writer: OutputWriter,
        show_progress: bool = True,
    ) -> int:
        """
        Run TAMMI analysis pipeline.
        
        Args:
            reader: Input source reader
            writer: Output destination writer
            show_progress: Whether to show progress bar
            
        Returns:
            Number of texts processed
        """
        total = reader.count()
        progress = ProgressBar(total) if show_progress else None
        
        # Write header
        writer.write_header(COLUMN_NAMES)
        
        # Process texts
        count = 0
        text_stream = reader.stream(lowercase=self.lowercase)
        
        for text_id, metrics in self.analyzer.analyze_texts(
            text_stream,
            batch_size=self.batch_size,
            n_process=self.n_process,
        ):
            writer.write_record(text_id, metrics)
            count += 1
            if progress:
                progress.update(count)
        
        if progress:
            progress.close()
        
        return count
    
    def run_with_batched_output(
        self,
        reader: InputReader,
        writer: OutputWriter,
        batch_write_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Run analysis with batched writes (better for database output).
        
        Args:
            reader: Input source reader
            writer: Output destination writer  
            batch_write_size: Number of records to batch before writing
            show_progress: Whether to show progress bar
            
        Returns:
            Number of texts processed
        """
        total = reader.count()
        progress = ProgressBar(total) if show_progress else None
        
        results_batch: List[Tuple[str, List[float]]] = []
        count = 0
        
        text_stream = reader.stream(lowercase=self.lowercase)
        
        for text_id, metrics in self.analyzer.analyze_texts(
            text_stream,
            batch_size=self.batch_size,
            n_process=self.n_process,
        ):
            results_batch.append((text_id, metrics))
            count += 1
            
            if len(results_batch) >= batch_write_size:
                writer.write_batch(results_batch)
                results_batch = []
            
            if progress:
                progress.update(count)
        
        # Write remaining
        if results_batch:
            writer.write_batch(results_batch)
        
        if progress:
            progress.close()
        
        return count
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TAMMIRunner":
        """Create runner from parsed CLI arguments."""
        return cls(
            morpholex_path=args.morpholex,
            spacy_model=args.model,
            use_gpu=getattr(args, "use_gpu", False),
            batch_size=args.batch_size,
            n_process=args.n_process,
            lowercase=not args.keep_case,
        )


def create_reader_from_args(args: argparse.Namespace) -> InputReader:
    """Create an InputReader from CLI arguments."""
    input_source_type = getattr(args, "input_source_type", "files")
    
    if input_source_type == "csv":
        from tammi.io.csv_io import CSVReader
        return CSVReader(
            path=args.input_csv_path,
            text_column=args.input_text_column,
            id_column=args.input_id_column,
        )
    
    elif input_source_type == "json":
        from tammi.io.json_io import JSONReader
        return JSONReader(
            path=args.input_json_path,
            text_column=args.input_text_column,
            id_column=args.input_id_column,
        )
    
    elif input_source_type in ("sqlite", "mysql", "postgresql", "mongodb"):
        db_config = args.input_db_config
        return ReaderFactory.create(input_source_type, db_config=db_config)
    
    else:  # files
        from tammi.io.file_io import TextFileReader
        extensions = tuple(ext.strip().lower() for ext in args.ext.split(",") if ext.strip())
        return TextFileReader(
            paths=args.inputs,
            extensions=extensions,
            recursive=args.recursive,
        )


def create_writer_from_args(args: argparse.Namespace) -> OutputWriter:
    """Create an OutputWriter from CLI arguments."""
    output_dest_type = getattr(args, "output_dest_type", "csv")
    
    if output_dest_type == "csv":
        from tammi.io.csv_io import CSVWriter
        return CSVWriter(path=args.output, columns=COLUMN_NAMES)
    
    elif output_dest_type == "json":
        from tammi.io.json_io import JSONWriter
        return JSONWriter(path=args.output, columns=COLUMN_NAMES)
    
    elif output_dest_type in ("sqlite", "mysql", "postgresql", "mongodb"):
        db_config = args.output_db_config
        return WriterFactory.create(output_dest_type, db_config=db_config, columns=COLUMN_NAMES)
    
    else:
        from tammi.io.csv_io import CSVWriter
        return CSVWriter(path=args.output, columns=COLUMN_NAMES)
