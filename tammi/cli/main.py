"""
Main CLI entry point for TAMMI.

TAMMI - Tool for Automatic Measurement of Morphological Information
CLI module refactored and enhanced by Ali Rezaei

Features:
- Multiple input formats: text files, CSV, JSON/JSONL, databases
- Multiple output formats: CSV, JSON, databases
- Database support: SQLite, MySQL, PostgreSQL, MongoDB
- GPU acceleration support
- Interactive menu mode
"""
from __future__ import annotations

__author__ = "Ali Rezaei"

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from tammi.analysis.metrics import COLUMN_NAMES
from tammi.io.base import AVAILABLE_DRIVERS
from tammi.io.database_io import DatabaseConfig
from tammi.cli.progress import quick_cpu_probe, check_gpu_available
from tammi.cli.menu import interactive_menu
from tammi.cli.runner import TAMMIRunner, create_reader_from_args, create_writer_from_args


DEFAULT_BATCH_SIZE = 1000
DEFAULT_N_PROCESS = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TAMMI on text files, CSV, JSON, or database and export morphological counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Text files or directories containing text files.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="en_core_web_sm",
        help="spaCy model name to use.",
    )
    parser.add_argument(
        "--morpholex",
        default="morpho_lex_df_w_log_w_prefsuf_no_head.csv",
        help="Path to the MorphoLex CSV (without header).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="morphemes.csv",
        help="Output CSV/JSON path.",
    )
    parser.add_argument(
        "--ext",
        default=".txt",
        help="Comma-separated list of file extensions to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when input paths are directories.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="spaCy pipe batch size.",
    )
    parser.add_argument(
        "--n-process",
        type=int,
        default=DEFAULT_N_PROCESS,
        help="Number of processes for spaCy pipe.",
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Do not lowercase input text (TAMMI originally lowercases).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration (requires CUDA or MPS). Automatically sets n_process=1.",
    )
    
    # Input source options
    parser.add_argument(
        "--input-csv",
        metavar="CSV_PATH",
        help="Use a CSV file as input source instead of text files.",
    )
    parser.add_argument(
        "--input-json",
        metavar="JSON_PATH",
        help="Use a JSON file as input source instead of text files.",
    )
    parser.add_argument(
        "--text-column",
        default="text_content",
        help="Column/field name containing text (for CSV/JSON/database input).",
    )
    parser.add_argument(
        "--id-column",
        default="text_id",
        help="Column/field name containing text IDs (for CSV/JSON/database input).",
    )
    parser.add_argument(
        "--input-db",
        metavar="CONNECTION",
        help="Database connection string for input. Format: sqlite:path.db, mysql://user:pass@host/db, postgresql://user:pass@host/db, mongodb://user:pass@host/db",
    )
    parser.add_argument(
        "--input-table",
        default="tammi_texts",
        help="Table/collection name for database input.",
    )
    
    # Output destination options
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output to JSON format instead of CSV.",
    )
    parser.add_argument(
        "--output-db",
        metavar="CONNECTION",
        help="Database connection string for output.",
    )
    parser.add_argument(
        "--output-table",
        default="tammi_results",
        help="Table/collection name for database output.",
    )
    
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Launch an interactive menu to enter options instead of passing arguments.",
    )
    
    args = parser.parse_args()
    
    # Determine input source type from arguments
    if args.input_csv:
        args.input_source_type = "csv"
        args.input_csv_path = args.input_csv
        args.input_json_path = ""
        args.input_text_column = args.text_column
        args.input_id_column = args.id_column
        args.input_db_config = None
    elif args.input_json:
        args.input_source_type = "json"
        args.input_json_path = args.input_json
        args.input_csv_path = ""
        args.input_text_column = args.text_column
        args.input_id_column = args.id_column
        args.input_db_config = None
    elif args.input_db:
        args.input_source_type, args.input_db_config = _parse_db_connection(
            args.input_db, args.input_table, args.text_column, args.id_column
        )
        args.input_csv_path = ""
        args.input_json_path = ""
        args.input_text_column = args.text_column
        args.input_id_column = args.id_column
    else:
        args.input_source_type = "files"
        args.input_csv_path = ""
        args.input_json_path = ""
        args.input_text_column = ""
        args.input_id_column = ""
        args.input_db_config = None
    
    # Determine output destination type
    if args.output_db:
        args.output_dest_type, args.output_db_config = _parse_db_connection(
            args.output_db, args.output_table, "", ""
        )
    elif args.output_json or args.output.endswith(".json"):
        args.output_dest_type = "json"
        args.output_db_config = None
    else:
        args.output_dest_type = "csv"
        args.output_db_config = None
    
    return args


def _parse_db_connection(
    conn_str: str, table: str, text_col: str, id_col: str
) -> tuple[str, DatabaseConfig]:
    """Parse a database connection string into a DatabaseConfig."""
    if conn_str.startswith("sqlite:"):
        db_path = conn_str[7:]
        return "sqlite", DatabaseConfig(
            db_type="sqlite",
            database=db_path,
            table=table,
            text_column=text_col or "text_content",
            id_column=id_col or "text_id",
        )
    elif conn_str.startswith(("mysql://", "postgresql://", "mongodb://")):
        parsed = urlparse(conn_str)
        db_type = parsed.scheme
        return db_type, DatabaseConfig(
            db_type=db_type,
            host=parsed.hostname or "localhost",
            port=parsed.port or DatabaseConfig._default_port(db_type),
            database=parsed.path.lstrip("/") if parsed.path else "",
            username=parsed.username or "",
            password=parsed.password or "",
            table=table,
            text_column=text_col or "text_content",
            id_column=id_col or "text_id",
        )
    else:
        raise SystemExit(
            f"Invalid database connection string: {conn_str}. "
            "Use sqlite:path.db, mysql://..., postgresql://..., or mongodb://..."
        )


def run_tammi(args: argparse.Namespace) -> None:
    """Run TAMMI analysis with the given arguments."""
    # Validate morpholex path
    morph_path = Path(args.morpholex)
    if not morph_path.exists():
        raise SystemExit(f"MorphoLex file not found: {morph_path}")
    
    # Handle GPU setup
    use_gpu = getattr(args, "use_gpu", False)
    if use_gpu:
        gpu_available, gpu_msg = check_gpu_available()
        print(f"GPU: {gpu_msg}")
        if gpu_available:
            if args.n_process != 1:
                print("  (Overriding n_process to 1 for GPU mode)")
                args.n_process = 1
        else:
            print("  (Falling back to CPU)")
            use_gpu = False
    
    # Create runner
    runner = TAMMIRunner(
        morpholex_path=args.morpholex,
        spacy_model=args.model,
        use_gpu=use_gpu,
        batch_size=args.batch_size,
        n_process=args.n_process,
        lowercase=not args.keep_case,
    )
    
    # Create reader and writer
    reader = create_reader_from_args(args)
    writer = create_writer_from_args(args)
    
    total = reader.count()
    input_desc = _get_input_description(args)
    print(f"Processing {total} items from {input_desc}")
    
    try:
        # Use batched output for database destinations
        output_dest_type = getattr(args, "output_dest_type", "csv")
        if output_dest_type in ("sqlite", "mysql", "postgresql", "mongodb"):
            count = runner.run_with_batched_output(reader, writer)
            output_desc = f"{output_dest_type} database: {args.output_db_config.table}"
        else:
            count = runner.run(reader, writer)
            output_desc = args.output
        
        print(f"Wrote {count} results to: {output_desc}")
    finally:
        reader.close()
        writer.close()


def _get_input_description(args: argparse.Namespace) -> str:
    """Get a human-readable description of the input source."""
    input_type = getattr(args, "input_source_type", "files")
    
    if input_type == "csv":
        return f"CSV: {args.input_csv_path}"
    elif input_type == "json":
        return f"JSON: {args.input_json_path}"
    elif input_type in ("sqlite", "mysql", "postgresql", "mongodb"):
        db_config = args.input_db_config
        return f"{input_type} database: {db_config.database}.{db_config.table}"
    else:
        return f"files: {', '.join(args.inputs)}"


def main() -> None:
    """Main CLI entry point."""
    suggested_n_process, suggested_batch, tokens_per_sec = quick_cpu_probe()
    if tokens_per_sec:
        print(
            f"Quick CPU check: ~{tokens_per_sec:,.0f} tokens/sec with a blank spaCy model."
        )
    cores = os.cpu_count() or 1
    print(
        f"Suggested settings -> batch_size: {suggested_batch}, n_process: {suggested_n_process} "
        f"(detected cores: {cores})"
    )

    args = parse_args()
    if args.batch_size == DEFAULT_BATCH_SIZE:
        args.batch_size = suggested_batch
    if args.n_process == DEFAULT_N_PROCESS:
        args.n_process = suggested_n_process

    # Check if we have any valid input source
    has_input = (
        args.inputs or 
        getattr(args, 'input_csv_path', '') or 
        getattr(args, 'input_json_path', '') or
        getattr(args, 'input_db_config', None)
    )
    
    if args.menu or not has_input:
        args = interactive_menu(args, suggested_n_process, suggested_batch)
    
    run_tammi(args)


if __name__ == "__main__":
    main()
