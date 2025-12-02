#!/usr/bin/env python3
"""
Command-line runner for TAMMI (Tool for Automatic Measurement of Morphological Information).

This script rewrites the notebook workflow into a reusable CLI that can process many text
files efficiently. It streams texts through spaCy, loads the MorphoLex CSV once, and writes
results straight to a CSV so very large batches don't require holding everything in memory.

Supports input/output via:
- CSV files
- Text files (.txt)
- SQLite databases
- MySQL databases
- PostgreSQL databases
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import spacy


# =============================================================================
# Database Support - Optional Imports
# =============================================================================

def _check_db_drivers() -> Dict[str, bool]:
    """Check which database drivers are available."""
    drivers = {}
    try:
        import sqlite3  # noqa: F401
        drivers["sqlite"] = True
    except ImportError:
        drivers["sqlite"] = False
    
    try:
        import mysql.connector  # noqa: F401
        drivers["mysql"] = True
    except ImportError:
        drivers["mysql"] = False
    
    try:
        import psycopg2  # noqa: F401
        drivers["postgresql"] = True
    except ImportError:
        drivers["postgresql"] = False
    
    return drivers


DB_DRIVERS = _check_db_drivers()


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    db_type: str  # 'sqlite', 'mysql', 'postgresql'
    host: str = "localhost"
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""
    table: str = "tammi_results"
    text_column: str = "text_content"
    id_column: str = "text_id"
    
    def get_connection(self):
        """Get a database connection based on config."""
        if self.db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(self.database)
        elif self.db_type == "mysql":
            import mysql.connector
            return mysql.connector.connect(
                host=self.host,
                port=self.port or 3306,
                database=self.database,
                user=self.username,
                password=self.password,
            )
        elif self.db_type == "postgresql":
            import psycopg2
            return psycopg2.connect(
                host=self.host,
                port=self.port or 5432,
                dbname=self.database,
                user=self.username,
                password=self.password,
            )
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


COLUMN_NAMES: List[str] = [
    "Inflected_Tokens",
    "Derivational_Tokens",
    "Tokens_w_Prefixes",
    "Tokens_w_Affixes",
    "Compounds",
    "number_prefixes",
    "number_roots",
    "number_suffixes",
    "number_affixes",
    "num_roots_affixes",
    "num_root_affix_inflec",
    "%_more_freq_words_morpho-family_prefix",
    "prefix_family_size",
    "prefix_freq",
    "prefix_log_freq",
    "prefix_len",
    "prefix_in_hapax",
    "hapax_in_prefix",
    "%_more_freq_words_morpho-family_root",
    "root_family_size",
    "root_freq",
    "root_log_freq",
    "%_more_freq_words_morpho-family_suffix",
    "suffix_family_size",
    "suffix_freq",
    "suffix_log_freq",
    "suffix_len",
    "suffix_in_hapax",
    "hapax_in_suffix",
    "%_more_freq_words_morpho-family_affix",
    "affix_family_size",
    "affix_freq",
    "affix_log_freq",
    "affix_len",
    "affix_in_hapax",
    "hapax_in_affix",
    "mean subset inflectional variety (10)",
    "inflectional TTR (10)",
    "inflectional MCI (10)",
    "mean subset derivational variety (10)",
    "derivational TTR (10)",
    "derivational MCI (10)",
]

# Indices in the MorphoLex CSV that list derivational affixes for a word.
DERIVATIONAL_AFFIX_INDICES: Tuple[int, ...] = (68, 69, 70, 71, 72, 73, 74)
DEFAULT_BATCH_SIZE = 1000
DEFAULT_N_PROCESS = 1


def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU acceleration is available for spaCy.
    Returns (is_available, message).
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA GPU available: {device_name}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "Apple MPS (Metal) GPU available"
        else:
            return False, "No GPU detected (torch installed but no CUDA/MPS)"
    except ImportError:
        pass

    try:
        import cupy
        return True, "CUDA GPU available via CuPy"
    except ImportError:
        pass

    return False, "No GPU support detected (install torch with CUDA or cupy)"


def setup_gpu(prefer_gpu: bool) -> Tuple[bool, str]:
    """
    Attempt to enable GPU for spaCy if requested.
    Returns (gpu_enabled, message).
    """
    if not prefer_gpu:
        return False, "GPU not requested"

    gpu_available, availability_msg = check_gpu_available()
    if not gpu_available:
        return False, f"GPU requested but not available: {availability_msg}"

    try:
        # Try to enable GPU in spaCy
        gpu_activated = spacy.prefer_gpu()
        if gpu_activated:
            return True, f"GPU enabled for spaCy. {availability_msg}"
        else:
            return False, "spacy.prefer_gpu() returned False - falling back to CPU"
    except Exception as e:
        return False, f"Failed to enable GPU: {e}"


class ProgressBar:
    """Lightweight ASCII progress bar for CLI runs."""

    def __init__(self, total: int, width: int = 30) -> None:
        self.total = total
        self.width = width
        self.enabled = sys.stdout.isatty() and total > 0

    def update(self, current: int) -> None:
        if not self.enabled:
            return
        ratio = min(max(current / self.total, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(
            f"\rProcessing texts [{bar}] {current}/{self.total} ({ratio * 100:5.1f}%)"
        )
        sys.stdout.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stdout.write("\n")


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def list_windows(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_morph_dict(path: Path) -> Dict[str, List]:
    morph_dict: Dict[str, List] = {}
    with path.open(newline="", encoding="utf-8") as inp:
        reader = csv.reader(inp)
        for row in reader:
            key, raw_values = row[0], row[1:76]
            values: List = []
            for val in raw_values:
                try:
                    values.append(float(val))
                except ValueError:
                    values.append(val)
            morph_dict[key] = values
    return morph_dict


# =============================================================================
# Input/Output Sources - Files and Databases
# =============================================================================

@dataclass
class IOConfig:
    """Configuration for input/output sources."""
    source_type: str  # 'files', 'csv', 'database'
    # For files
    paths: List[str] = None  # type: ignore
    extensions: Tuple[str, ...] = (".txt",)
    recursive: bool = False
    # For CSV input
    csv_path: str = ""
    text_column: str = "text"
    id_column: str = "id"
    # For database
    db_config: Optional[DatabaseConfig] = None
    
    def __post_init__(self):
        if self.paths is None:
            self.paths = []


def _discover_csv_files(base_path: Path = Path(".")) -> List[Path]:
    """Find CSV files that could be used as input/output."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            candidates.append(item)
    return sorted(candidates)


def stream_texts_from_csv(
    csv_path: Path, 
    text_column: str, 
    id_column: str, 
    lowercase: bool
) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Stream texts from a CSV file."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column, "")
            text_id = row.get(id_column, "")
            if lowercase:
                text = text.lower()
            yield text, {"text_id": text_id}


def stream_texts_from_database(
    db_config: DatabaseConfig, 
    lowercase: bool
) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Stream texts from a database table."""
    conn = db_config.get_connection()
    cursor = conn.cursor()
    
    query = f"SELECT {db_config.id_column}, {db_config.text_column} FROM {db_config.table}"
    cursor.execute(query)
    
    for row in cursor:
        text_id, text = row[0], row[1] or ""
        if lowercase:
            text = text.lower()
        yield text, {"text_id": str(text_id)}
    
    cursor.close()
    conn.close()


def write_results_to_database(
    db_config: DatabaseConfig,
    results: List[Tuple[str, List[float]]],
) -> int:
    """Write TAMMI results to a database table."""
    conn = db_config.get_connection()
    cursor = conn.cursor()
    
    # Create table if not exists - quote column names for special characters
    columns_sql = ", ".join([f'"{col}" REAL' for col in COLUMN_NAMES])
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS "{db_config.table}" (
            text_id TEXT PRIMARY KEY,
            {columns_sql}
        )
    """
    cursor.execute(create_sql)
    
    # Insert results - quote column names for special characters
    quoted_columns = ", ".join([f'"{col}"' for col in COLUMN_NAMES])
    placeholders = ", ".join(["?" if db_config.db_type == "sqlite" else "%s"] * (len(COLUMN_NAMES) + 1))
    insert_sql = f'INSERT INTO "{db_config.table}" (text_id, {quoted_columns}) VALUES ({placeholders})'
    
    for text_id, values in results:
        cursor.execute(insert_sql, [text_id] + values)
    
    conn.commit()
    count = len(results)
    cursor.close()
    conn.close()
    return count


def get_csv_columns(csv_path: Path) -> List[str]:
    """Get column names from a CSV file."""
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def iter_text_paths(
    inputs: List[str], extensions: Tuple[str, ...], recursive: bool
) -> Iterator[Path]:
    for item in inputs:
        path = Path(item)
        if path.is_file():
            if path.suffix.lower() in extensions:
                yield path
            continue
        if not path.is_dir():
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in extensions:
                yield candidate


def stream_texts(
    paths: Iterable[Path], lowercase: bool
) -> Iterator[Tuple[str, Dict[str, str]]]:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        if lowercase:
            text = text.lower()
        yield text, {"text_id": str(path)}


def calc_mci(msv: float, between_unique: int, subset_count: int) -> float:
    return safe_divide(msv + between_unique, subset_count) - 1 if subset_count else 0.0


def analyze_doc(doc, morph_dict: Dict[str, List]) -> List[float]:
    cw: List[str] = []
    final_result_list_cw: List[float] = []

    tokens_w_inflections: List[int] = []
    tokens_w_derivations: List[int] = []
    tokens_w_prefixes: List[int] = []
    tokens_w_suffixes: List[int] = []
    compounds: List[int] = []
    num_prefix: List[float] = []
    num_root: List[float] = []
    num_suffix: List[float] = []
    num_affix: List[float] = []
    num_root_affix: List[float] = []
    prefix_PFMF: List[float] = []
    prefix_fam_size: List[float] = []
    prefix_freq: List[float] = []
    prefix_log_freq: List[float] = []
    prefix_len: List[float] = []
    prefix_in_hapax: List[float] = []
    hapax_in_prefix: List[float] = []
    root_PFMF: List[float] = []
    root_fam_size: List[float] = []
    root_freq: List[float] = []
    root_log_freq: List[float] = []
    suffix_PFMF: List[float] = []
    suffix_fam_size: List[float] = []
    suffix_freq: List[float] = []
    suffix_log_freq: List[float] = []
    suffix_len: List[float] = []
    suffix_in_hapax: List[float] = []
    hapax_in_suffix: List[float] = []
    affix_PFMF: List[float] = []
    affix_fam_size: List[float] = []
    affix_freq: List[float] = []
    affix_log_freq: List[float] = []
    affix_len: List[float] = []
    affix_in_hapax: List[float] = []
    hapax_in_affix: List[float] = []
    inflections: List[str] = []

    for token in doc:
        if token.is_stop or not token.is_alpha:
            continue
        text = token.text
        lemma = token.lemma_

        cw.append(text)
        inflections.append(text.replace(lemma, "") if len(text) > len(lemma) else "")
        if text != lemma:
            tokens_w_inflections.append(1)

        val = morph_dict.get(text)
        if val is None:
            continue

        prefix_count, root_count, suffix_count = val[3], val[4], val[5]
        if prefix_count >= 1 or suffix_count >= 1:
            tokens_w_derivations.append(1)
        if prefix_count >= 1:
            tokens_w_prefixes.append(1)
        if suffix_count >= 1:
            tokens_w_suffixes.append(1)
        if root_count > 1:
            compounds.append(1)

        num_prefix.append(prefix_count)
        num_root.append(root_count)
        num_suffix.append(suffix_count)
        num_affix.extend([prefix_count, suffix_count])
        num_root_affix.extend([root_count, prefix_count, suffix_count])

        prefix_PFMF.extend([val[7], val[14], val[21]])
        prefix_fam_size.extend([val[8], val[15], val[22]])
        prefix_freq.extend([val[9], val[16], val[23]])
        prefix_log_freq.extend([val[10], val[17], val[24]])
        prefix_len.extend([val[11], val[18], val[25]])
        prefix_in_hapax.extend([val[12], val[19], val[26]])
        hapax_in_prefix.extend([val[13], val[20], val[27]])

        root_PFMF.extend([val[28], val[32], val[36]])
        root_fam_size.extend([val[29], val[33], val[37]])
        root_freq.extend([val[30], val[34], val[38]])
        root_log_freq.extend([val[31], val[35], val[39]])

        suffix_PFMF.extend([val[40], val[47], val[54], val[61]])
        suffix_fam_size.extend([val[41], val[48], val[55], val[62]])
        suffix_freq.extend([val[42], val[49], val[56], val[63]])
        suffix_log_freq.extend([val[43], val[50], val[57], val[64]])
        suffix_len.extend([val[44], val[51], val[58], val[65]])
        suffix_in_hapax.extend([val[45], val[52], val[59], val[66]])
        hapax_in_suffix.extend([val[46], val[53], val[60], val[67]])

        affix_PFMF.extend([val[7], val[14], val[21], val[40], val[47], val[54], val[61]])
        affix_fam_size.extend(
            [val[8], val[15], val[22], val[41], val[48], val[55], val[62]]
        )
        affix_freq.extend(
            [val[9], val[16], val[23], val[42], val[49], val[56], val[63]]
        )
        affix_log_freq.extend(
            [val[10], val[17], val[24], val[43], val[50], val[57], val[64]]
        )
        affix_len.extend(
            [val[11], val[18], val[25], val[44], val[51], val[58], val[65]]
        )
        affix_in_hapax.extend(
            [val[12], val[19], val[26], val[45], val[52], val[59], val[66]]
        )
        hapax_in_affix.extend(
            [val[13], val[20], val[27], val[46], val[53], val[60], val[67]]
        )

    length = len(cw)

    final_result_list_cw.append(safe_divide(len(tokens_w_inflections), length))
    final_result_list_cw.append(safe_divide(len(tokens_w_derivations), length))
    final_result_list_cw.append(safe_divide(len(tokens_w_prefixes), length))
    final_result_list_cw.append(safe_divide(len(tokens_w_suffixes), length))
    final_result_list_cw.append(safe_divide(len(compounds), length))

    final_result_list_cw.append(safe_divide(sum(num_prefix), length))
    final_result_list_cw.append(safe_divide(sum(num_root), length))
    final_result_list_cw.append(safe_divide(sum(num_suffix), length))
    final_result_list_cw.append(safe_divide(sum(num_affix), length))
    final_result_list_cw.append(safe_divide(sum(num_root_affix), length))

    num_root_affix_inflec = tokens_w_inflections + num_root_affix
    final_result_list_cw.append(safe_divide(len(num_root_affix_inflec), length))

    def cal_fixes(fix_var: List) -> None:
        cleaned = [x for x in fix_var if x != "" and x is not None]
        final_result_list_cw.append(safe_divide(sum(cleaned), length))

    cal_fixes(prefix_PFMF)
    cal_fixes(prefix_fam_size)
    cal_fixes(prefix_freq)
    cal_fixes(prefix_log_freq)
    cal_fixes(prefix_len)
    cal_fixes(prefix_in_hapax)
    cal_fixes(hapax_in_prefix)
    cal_fixes(root_PFMF)
    cal_fixes(root_fam_size)
    cal_fixes(root_freq)
    cal_fixes(root_log_freq)
    cal_fixes(suffix_PFMF)
    cal_fixes(suffix_fam_size)
    cal_fixes(suffix_freq)
    cal_fixes(suffix_log_freq)
    cal_fixes(suffix_len)
    cal_fixes(suffix_in_hapax)
    cal_fixes(hapax_in_suffix)
    cal_fixes(affix_PFMF)
    cal_fixes(affix_fam_size)
    cal_fixes(affix_freq)
    cal_fixes(affix_log_freq)
    cal_fixes(affix_len)
    cal_fixes(affix_in_hapax)
    cal_fixes(hapax_in_affix)

    mw10 = list(list_windows(inflections, 10))
    inflec_10_types: List[int] = [len(set(w)) - 1 for w in mw10]
    inflec_10_tokens: List[int] = [len(w) for w in mw10]
    inflec_10_subset: List[int] = [len(set(w)) for w in mw10]

    ttr_10_inflec = [
        safe_divide(types, tokens)
        for types, tokens in zip(inflec_10_types, inflec_10_tokens)
    ]
    inflec_10_ttr = safe_divide(sum(ttr_10_inflec), len(ttr_10_inflec))
    msv_10_inflec = safe_divide(sum(inflec_10_subset), len(inflec_10_subset))

    sing_list_inflec: List[str] = list(chain.from_iterable(mw10))
    inflec_counts = Counter(sing_list_inflec)
    between_subset_div_inflec = sum(1 for v in inflec_counts.values() if v == 1)
    mci_inflec = calc_mci(msv_10_inflec, between_subset_div_inflec, len(inflec_10_subset))

    final_result_list_cw.append(msv_10_inflec)
    final_result_list_cw.append(inflec_10_ttr)
    final_result_list_cw.append(mci_inflec)

    cw10 = list(list_windows(cw, 10))
    cw10_affixes: List[List[str]] = []
    for window in cw10:
        affix_per_window: List[str] = []
        for word in window:
            val = morph_dict.get(word)
            if val is None:
                continue
            affix_per_window.extend(
                aff for aff in (val[i] for i in DERIVATIONAL_AFFIX_INDICES) if aff
            )
        cw10_affixes.append(affix_per_window)

    deriv_10_types = [len(set(cw10_aff_el)) for cw10_aff_el in cw10_affixes]
    deriv_10_tokens = [len(cw10_aff_el) for cw10_aff_el in cw10_affixes]
    ttr_10_deriv = [
        safe_divide(types, tokens)
        for types, tokens in zip(deriv_10_types, deriv_10_tokens)
    ]
    deriv_10_ttr = safe_divide(sum(ttr_10_deriv), len(ttr_10_deriv))
    msv_10_deriv = safe_divide(sum(deriv_10_types), len(deriv_10_types))

    sing_list_der = list(chain.from_iterable(cw10_affixes))
    deriv_counts = Counter(sing_list_der)
    between_subset_div_der = sum(1 for v in deriv_counts.values() if v == 1)
    mci_deriv = calc_mci(msv_10_deriv, between_subset_div_der, len(deriv_10_types))

    final_result_list_cw.append(msv_10_deriv)
    final_result_list_cw.append(deriv_10_ttr)
    final_result_list_cw.append(mci_deriv)

    return final_result_list_cw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAMMI on text files, CSV, or database and export morphological counts.",
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
        help="Output CSV path.",
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
        "--text-column",
        default="text_content",
        help="Column name containing text (for CSV input).",
    )
    parser.add_argument(
        "--id-column",
        default="text_id",
        help="Column name containing text IDs (for CSV input).",
    )
    parser.add_argument(
        "--input-db",
        metavar="CONNECTION",
        help="Database connection string for input. Format: sqlite:path.db or mysql://user:pass@host/db or postgresql://user:pass@host/db",
    )
    parser.add_argument(
        "--input-table",
        default="tammi_texts",
        help="Table name for database input.",
    )
    
    # Output destination options
    parser.add_argument(
        "--output-db",
        metavar="CONNECTION",
        help="Database connection string for output. Format: sqlite:path.db or mysql://user:pass@host/db or postgresql://user:pass@host/db",
    )
    parser.add_argument(
        "--output-table",
        default="tammi_results",
        help="Table name for database output.",
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
        args.input_text_column = args.text_column
        args.input_id_column = args.id_column
        args.input_db_config = None
    elif args.input_db:
        args.input_source_type, args.input_db_config = _parse_db_connection(args.input_db, args.input_table, args.text_column, args.id_column)
        args.input_csv_path = ""
        args.input_text_column = args.text_column
        args.input_id_column = args.id_column
    else:
        args.input_source_type = "files"
        args.input_csv_path = ""
        args.input_text_column = ""
        args.input_id_column = ""
        args.input_db_config = None
    
    # Determine output destination type
    if args.output_db:
        args.output_dest_type, args.output_db_config = _parse_db_connection(args.output_db, args.output_table, "", "")
    else:
        args.output_dest_type = "csv"
        args.output_db_config = None
    
    return args


def _parse_db_connection(conn_str: str, table: str, text_col: str, id_col: str) -> Tuple[str, DatabaseConfig]:
    """Parse a database connection string into a DatabaseConfig."""
    if conn_str.startswith("sqlite:"):
        db_path = conn_str[7:]  # Remove 'sqlite:'
        return "sqlite", DatabaseConfig(
            db_type="sqlite",
            database=db_path,
            table=table,
            text_column=text_col or "text_content",
            id_column=id_col or "text_id",
        )
    elif conn_str.startswith("mysql://") or conn_str.startswith("postgresql://"):
        parsed = urlparse(conn_str)
        db_type = "mysql" if conn_str.startswith("mysql://") else "postgresql"
        return db_type, DatabaseConfig(
            db_type=db_type,
            host=parsed.hostname or "localhost",
            port=parsed.port or (3306 if db_type == "mysql" else 5432),
            database=parsed.path.lstrip("/") if parsed.path else "",
            username=parsed.username or "",
            password=parsed.password or "",
            table=table,
            text_column=text_col or "text_content",
            id_column=id_col or "text_id",
        )
    else:
        raise SystemExit(f"Invalid database connection string: {conn_str}. Use sqlite:path.db, mysql://..., or postgresql://...")


def _prompt_with_default(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def _prompt_bool(prompt: str, default: bool) -> bool:
    default_label = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n) [{default_label}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number.")


# =============================================================================
# Enhanced Interactive Menu Helpers
# =============================================================================

def _clear_line() -> None:
    """Clear the current line in terminal."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _print_menu_header(title: str) -> None:
    """Print a styled menu section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def _prompt_choice(
    prompt: str,
    options: List[str],
    default_index: int = 0,
    allow_custom: bool = False,
    custom_label: str = "Enter custom value",
) -> Tuple[int, str]:
    """
    Display a numbered menu and let user pick an option.
    Returns (selected_index, selected_value).
    If allow_custom and user picks custom, returns (-1, custom_value).
    """
    print(f"\n  {prompt}")
    for i, opt in enumerate(options):
        marker = "→" if i == default_index else " "
        print(f"    {marker} [{i + 1}] {opt}")
    if allow_custom:
        print(f"      [{len(options) + 1}] {custom_label}")

    max_choice = len(options) + (1 if allow_custom else 0)
    while True:
        hint = f"1-{max_choice}" if max_choice > 1 else "1"
        raw = input(f"  Enter choice ({hint}) [default: {default_index + 1}]: ").strip()
        if not raw:
            return default_index, options[default_index]
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice - 1, options[choice - 1]
            if allow_custom and choice == len(options) + 1:
                custom_val = input(f"  {custom_label}: ").strip()
                return -1, custom_val
            print(f"  Please enter a number between 1 and {max_choice}.")
        except ValueError:
            print(f"  Invalid input. Enter a number between 1 and {max_choice}.")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Display a Yes/No choice menu.
    """
    options = ["Yes", "No"]
    default_idx = 0 if default else 1
    idx, _ = _prompt_choice(prompt, options, default_index=default_idx)
    return idx == 0


def _prompt_int_with_suggestions(
    prompt: str,
    suggestions: List[int],
    default: int,
    allow_custom: bool = True,
) -> int:
    """
    Display suggested integer values as choices.
    """
    options = [str(s) for s in suggestions]
    try:
        default_idx = suggestions.index(default)
    except ValueError:
        default_idx = 0

    idx, val = _prompt_choice(
        prompt,
        options,
        default_index=default_idx,
        allow_custom=allow_custom,
        custom_label="Enter custom number",
    )
    if idx == -1:
        # Custom value entered
        try:
            return int(val)
        except ValueError:
            print(f"  Invalid number, using default: {default}")
            return default
    return suggestions[idx]


def _discover_input_folders(base_path: Path = Path(".")) -> List[Path]:
    """
    Find directories that likely contain text files for processing.
    """
    candidates = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith((".", "_")):
            # Check if it has any .txt files
            txt_files = list(item.glob("*.txt"))[:5]  # Sample check
            if txt_files:
                candidates.append(item)
    return sorted(candidates)


def _discover_morpholex_files(base_path: Path = Path(".")) -> List[Path]:
    """
    Find potential MorphoLex CSV files in the current directory.
    """
    candidates = []
    for item in base_path.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            if "morph" in item.name.lower() or "lex" in item.name.lower():
                candidates.append(item)
    return sorted(candidates)


def _discover_spacy_models() -> List[str]:
    """
    Find installed spaCy models.
    """
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "info", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            pipelines = info.get("pipelines", {})
            if pipelines:
                return list(pipelines.keys())
    except Exception:
        pass
    # Fallback to common models
    return ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]


def quick_cpu_probe(sample_size: int = 2000) -> Tuple[int, int, float]:
    """
    Run a very small benchmark to suggest n_process and batch_size.
    """

    cores = os.cpu_count() or 1
    suggested_n_process = max(1, cores - 1)

    try:
        nlp = spacy.blank("en")
        sample_text = "This is a quick TAMMI benchmark sentence."
        sample = [sample_text] * sample_size
        start = time.perf_counter()
        token_count = 0
        for doc in nlp.pipe(sample, batch_size=100, n_process=1):
            token_count += len(doc)
        duration = max(time.perf_counter() - start, 1e-9)
        tokens_per_sec = token_count / duration
    except Exception:
        tokens_per_sec = 0.0

    if tokens_per_sec >= 40000:
        suggested_batch = 2000
    elif tokens_per_sec >= 20000:
        suggested_batch = 1500
    elif tokens_per_sec >= 10000:
        suggested_batch = 1000
    else:
        suggested_batch = 500

    return suggested_n_process, suggested_batch, tokens_per_sec


def interactive_menu(
    base_args: argparse.Namespace,
    suggested_n_process: int,
    suggested_batch: int,
) -> argparse.Namespace:
    """
    Enhanced interactive menu with numbered choices for easier use.
    Supports files, CSV, and database input/output.
    """
    print("\n" + "=" * 60)
    print("  🔬 TAMMI - Tool for Automatic Measurement of Morphological Info")
    print("=" * 60)
    print("  Use number keys to select options. Press Enter for defaults.\n")

    # -------------------------------------------------------------------------
    # 1. INPUT SOURCE TYPE SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("📁 Step 1: Select Input Source")
    
    # Build input source options based on available drivers
    input_source_options = [
        "Text files (.txt) from folder",
        "CSV file (with text column)",
    ]
    input_source_types = ["files", "csv"]
    
    if DB_DRIVERS.get("sqlite"):
        input_source_options.append("SQLite database")
        input_source_types.append("sqlite")
    if DB_DRIVERS.get("mysql"):
        input_source_options.append("MySQL database")
        input_source_types.append("mysql")
    if DB_DRIVERS.get("postgresql"):
        input_source_options.append("PostgreSQL database")
        input_source_types.append("postgresql")
    
    idx, _ = _prompt_choice("Select input source type:", input_source_options, default_index=0)
    input_source_type = input_source_types[idx]
    
    # Variables to hold input config
    inputs = []
    input_csv_path = ""
    input_text_column = "text"
    input_id_column = "id"
    input_db_config: Optional[DatabaseConfig] = None
    ext = ".txt"
    recursive = False
    
    if input_source_type == "files":
        # Discover available folders with text files
        available_folders = _discover_input_folders()
        
        if available_folders:
            folder_options = [f"{f.name}/ ({len(list(f.glob('*.txt')))} .txt files)" for f in available_folders]
            folder_options.append("Enter custom path")
            
            print("\n  Found folders with text files:")
            idx, _ = _prompt_choice(
                "Select input folder:",
                folder_options,
                default_index=0,
            )
            
            if idx == len(available_folders):
                raw_inputs = input("\n  Enter file/folder path: ").strip()
                inputs = [raw_inputs] if raw_inputs else []
            else:
                inputs = [str(available_folders[idx])]
        else:
            raw_inputs = input("  Enter text files or directory path: ").strip()
            inputs = [raw_inputs] if raw_inputs else []
        
        if not inputs:
            print("  ⚠️  No input path provided, using current directory.")
            inputs = ["."]
        
        # File extensions
        ext_options = [".txt", ".txt,.md", ".txt,.text", ".txt,.md,.rst"]
        idx, ext = _prompt_choice(
            "File extensions to process:",
            ext_options,
            default_index=0,
            allow_custom=True,
            custom_label="Enter custom extensions (comma-separated)",
        )
        if idx >= 0:
            ext = ext_options[idx]
        
        # Recursive search
        recursive = _prompt_yes_no("Search subdirectories recursively?", default=False)
    
    elif input_source_type == "csv":
        # CSV input
        available_csvs = _discover_csv_files()
        if available_csvs:
            csv_options = [str(f) for f in available_csvs]
            csv_options.append("Enter custom CSV path")
            
            idx, _ = _prompt_choice("Select input CSV file:", csv_options, default_index=0)
            if idx == len(available_csvs):
                input_csv_path = input("\n  Enter CSV file path: ").strip()
            else:
                input_csv_path = csv_options[idx]
        else:
            input_csv_path = input("  Enter CSV file path: ").strip()
        
        # Get columns from CSV
        if input_csv_path and Path(input_csv_path).exists():
            columns = get_csv_columns(Path(input_csv_path))
            if columns:
                print(f"\n  Found columns: {', '.join(columns)}")
                
                # Text column selection
                idx, input_text_column = _prompt_choice(
                    "Select column containing text:",
                    columns,
                    default_index=0,
                    allow_custom=True,
                    custom_label="Enter column name",
                )
                if idx >= 0:
                    input_text_column = columns[idx]
                
                # ID column selection
                idx, input_id_column = _prompt_choice(
                    "Select column containing text ID:",
                    columns,
                    default_index=0,
                    allow_custom=True,
                    custom_label="Enter column name",
                )
                if idx >= 0:
                    input_id_column = columns[idx]
            else:
                input_text_column = input("  Enter text column name [text]: ").strip() or "text"
                input_id_column = input("  Enter ID column name [id]: ").strip() or "id"
        
        inputs = [input_csv_path]
    
    else:
        # Database input
        input_db_config = _configure_database(input_source_type, "input")
        inputs = [f"db:{input_source_type}:{input_db_config.database}"]

    # -------------------------------------------------------------------------
    # 2. OUTPUT DESTINATION SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("💾 Step 2: Select Output Destination")
    
    output_dest_options = ["CSV file"]
    output_dest_types = ["csv"]
    
    if DB_DRIVERS.get("sqlite"):
        output_dest_options.append("SQLite database")
        output_dest_types.append("sqlite")
    if DB_DRIVERS.get("mysql"):
        output_dest_options.append("MySQL database")
        output_dest_types.append("mysql")
    if DB_DRIVERS.get("postgresql"):
        output_dest_options.append("PostgreSQL database")
        output_dest_types.append("postgresql")
    
    idx, _ = _prompt_choice("Select output destination:", output_dest_options, default_index=0)
    output_dest_type = output_dest_types[idx]
    
    output = "morphemes.csv"
    output_db_config: Optional[DatabaseConfig] = None
    
    if output_dest_type == "csv":
        # CSV output
        output_suggestions = ["morphemes.csv", "tammi_results.csv", "output/results.csv"]
        
        # Add input-based suggestion
        if inputs and not inputs[0].startswith("db:"):
            input_name = Path(inputs[0]).stem
            output_suggestions.insert(0, f"{input_name}_morphemes.csv")
        
        # Also show existing CSV files as options
        existing_csvs = _discover_csv_files()
        for csv_file in existing_csvs[:3]:
            if str(csv_file) not in output_suggestions:
                output_suggestions.append(str(csv_file))
        
        idx, output = _prompt_choice(
            "Select output filename:",
            output_suggestions,
            default_index=0,
            allow_custom=True,
            custom_label="Enter custom filename/path",
        )
        if idx >= 0:
            output = output_suggestions[idx]
    else:
        # Database output
        output_db_config = _configure_database(output_dest_type, "output")
        output = f"db:{output_dest_type}:{output_db_config.database}:{output_db_config.table}"

    # -------------------------------------------------------------------------
    # 3. SPACY MODEL SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("🧠 Step 3: Select spaCy Model")
    
    available_models = _discover_spacy_models()
    model_descriptions = {
        "en_core_web_sm": "Small (fast, ~12MB)",
        "en_core_web_md": "Medium (balanced, ~40MB)", 
        "en_core_web_lg": "Large (accurate, ~560MB)",
        "en_core_web_trf": "Transformer (most accurate, ~500MB, GPU recommended)",
    }
    
    model_options = []
    for m in available_models:
        desc = model_descriptions.get(m, "")
        model_options.append(f"{m} {f'- {desc}' if desc else ''}")
    
    try:
        default_model_idx = available_models.index(base_args.model)
    except ValueError:
        default_model_idx = 0
    
    idx, selected_value = _prompt_choice(
        "Select spaCy model:",
        model_options,
        default_index=default_model_idx,
        allow_custom=True,
        custom_label="Enter different model name",
    )
    model = available_models[idx] if idx >= 0 else selected_value

    # -------------------------------------------------------------------------
    # 4. MORPHOLEX FILE SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("📖 Step 4: Select MorphoLex Dictionary")
    
    morpholex_files = _discover_morpholex_files()
    if morpholex_files:
        morph_options = [str(f) for f in morpholex_files]
        morph_options.append("Enter custom path")
        
        try:
            default_morph_idx = morph_options.index(base_args.morpholex)
        except ValueError:
            default_morph_idx = 0
        
        idx, morpholex = _prompt_choice(
            "Select MorphoLex CSV file:",
            morph_options,
            default_index=default_morph_idx,
        )
        if idx == len(morpholex_files):
            morpholex = input("\n  Enter MorphoLex CSV path: ").strip() or base_args.morpholex
        else:
            morpholex = morph_options[idx]
    else:
        morpholex = input(f"  MorphoLex CSV path [{base_args.morpholex}]: ").strip() or base_args.morpholex

    # -------------------------------------------------------------------------
    # 5. PROCESSING OPTIONS
    # -------------------------------------------------------------------------
    _print_menu_header("⚙️  Step 5: Processing Options")
    
    case_options = ["Lowercase text (recommended for TAMMI)", "Keep original case"]
    idx, _ = _prompt_choice("Text case handling:", case_options, default_index=0)
    keep_case = (idx == 1)

    # -------------------------------------------------------------------------
    # 6. PERFORMANCE SETTINGS
    # -------------------------------------------------------------------------
    _print_menu_header("🚀 Step 6: Performance Settings")
    
    gpu_available, gpu_msg = check_gpu_available()
    cores = os.cpu_count() or 1
    
    if gpu_available:
        print(f"\n  ✅ {gpu_msg}")
        processing_options = [
            f"CPU multi-processing ({suggested_n_process} processes recommended)",
            "GPU acceleration (single process, good for large docs)",
        ]
        idx, _ = _prompt_choice("Select processing mode:", processing_options, default_index=0)
        use_gpu = (idx == 1)
    else:
        print(f"\n  ℹ️  GPU not available: {gpu_msg}")
        print("  Using CPU multi-processing.")
        use_gpu = False

    # Batch size
    batch_suggestions = [500, 1000, 1500, 2000, 5000]
    if suggested_batch not in batch_suggestions:
        batch_suggestions.insert(0, suggested_batch)
    batch_suggestions = sorted(set(batch_suggestions))
    
    batch_size = _prompt_int_with_suggestions(
        f"Batch size (suggested: {suggested_batch}):",
        batch_suggestions,
        default=suggested_batch,
    )

    # N_PROCESS (only if not GPU)
    if use_gpu:
        print("\n  ℹ️  GPU mode: using single process")
        n_process = 1
    else:
        process_suggestions = [1, 2, 4, max(1, cores // 2), max(1, cores - 1)]
        process_suggestions = sorted(set(process_suggestions))
        
        n_process = _prompt_int_with_suggestions(
            f"Number of CPU processes (cores detected: {cores}):",
            process_suggestions,
            default=suggested_n_process,
        )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    _print_menu_header("📋 Configuration Summary")
    
    input_summary = ', '.join(inputs) if inputs else "None"
    if input_source_type == "csv":
        input_summary = f"CSV: {input_csv_path} (text: {input_text_column}, id: {input_id_column})"
    elif input_db_config:
        input_summary = f"DB: {input_source_type} - {input_db_config.database}.{input_db_config.table}"
    
    output_summary = output
    if output_db_config:
        output_summary = f"DB: {output_dest_type} - {output_db_config.database}.{output_db_config.table}"
    
    print(f"""
  Input:        {input_summary}
  Output:       {output_summary}
  Model:        {model}
  MorphoLex:    {morpholex}
  Extensions:   {ext}
  Recursive:    {'Yes' if recursive else 'No'}
  Keep case:    {'Yes' if keep_case else 'No'}
  GPU:          {'Yes' if use_gpu else 'No'}
  Batch size:   {batch_size}
  Processes:    {n_process}
""")
    
    if not _prompt_yes_no("Proceed with these settings?", default=True):
        print("\n  Cancelled. Run again to reconfigure.")
        raise SystemExit(0)

    return argparse.Namespace(
        inputs=inputs,
        input_source_type=input_source_type,
        input_csv_path=input_csv_path,
        input_text_column=input_text_column,
        input_id_column=input_id_column,
        input_db_config=input_db_config,
        output=output,
        output_dest_type=output_dest_type,
        output_db_config=output_db_config,
        model=model,
        morpholex=morpholex,
        ext=ext,
        recursive=recursive,
        keep_case=keep_case,
        use_gpu=use_gpu,
        batch_size=batch_size,
        n_process=n_process,
        menu=False,
    )


def _configure_database(db_type: str, purpose: str) -> DatabaseConfig:
    """Interactive configuration for database connection."""
    print(f"\n  Configure {db_type.upper()} database for {purpose}:")
    
    if db_type == "sqlite":
        db_path = input("  Database file path [tammi.db]: ").strip() or "tammi.db"
        table = input("  Table name [tammi_texts]: ").strip() or "tammi_texts"
        text_col = input("  Text column name [text_content]: ").strip() or "text_content"
        id_col = input("  ID column name [text_id]: ").strip() or "text_id"
        
        return DatabaseConfig(
            db_type="sqlite",
            database=db_path,
            table=table,
            text_column=text_col,
            id_column=id_col,
        )
    else:
        # MySQL or PostgreSQL
        host = input("  Host [localhost]: ").strip() or "localhost"
        port_str = input(f"  Port [{'3306' if db_type == 'mysql' else '5432'}]: ").strip()
        port = int(port_str) if port_str else (3306 if db_type == "mysql" else 5432)
        database = input("  Database name: ").strip()
        username = input("  Username: ").strip()
        password = input("  Password: ").strip()
        table = input("  Table name [tammi_texts]: ").strip() or "tammi_texts"
        text_col = input("  Text column name [text_content]: ").strip() or "text_content"
        id_col = input("  ID column name [text_id]: ").strip() or "text_id"
        
        return DatabaseConfig(
            db_type=db_type,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            table=table,
            text_column=text_col,
            id_column=id_col,
        )


def run_tammi(args: argparse.Namespace) -> None:
    extensions = tuple(ext.strip().lower() for ext in args.ext.split(",") if ext.strip())

    morph_path = Path(args.morpholex)
    if not morph_path.exists():
        raise SystemExit(f"MorphoLex file not found: {morph_path}")
    morph_dict = load_morph_dict(morph_path)

    # Handle GPU setup
    use_gpu = getattr(args, "use_gpu", False)
    if use_gpu:
        gpu_enabled, gpu_msg = setup_gpu(prefer_gpu=True)
        print(f"GPU: {gpu_msg}")
        if gpu_enabled:
            # GPU mode requires n_process=1
            if args.n_process != 1:
                print("  (Overriding n_process to 1 for GPU mode)")
                args.n_process = 1
        else:
            print("  (Falling back to CPU)")

    try:
        nlp = spacy.load(args.model)
    except OSError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            f"spaCy model '{args.model}' is not installed. "
            f"Install it with: python -m spacy download {args.model}"
        ) from exc

    # Determine input source type
    input_source_type = getattr(args, "input_source_type", "files")
    lowercase = not args.keep_case
    
    # Set up the text stream based on input source type
    if input_source_type == "csv":
        # CSV file input
        csv_path = Path(getattr(args, "input_csv_path", args.inputs[0] if args.inputs else ""))
        if not csv_path.exists():
            raise SystemExit(f"Input CSV file not found: {csv_path}")
        
        text_column = getattr(args, "input_text_column", "text")
        id_column = getattr(args, "input_id_column", "id")
        
        # Count rows for progress bar
        with csv_path.open(newline="", encoding="utf-8") as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        
        doc_stream = stream_texts_from_csv(csv_path, text_column, id_column, lowercase)
        progress: Optional[ProgressBar] = ProgressBar(total_rows)
        print(f"Processing {total_rows} rows from CSV: {csv_path}")
        
    elif input_source_type in ("sqlite", "mysql", "postgresql"):
        # Database input
        input_db_config: Optional[DatabaseConfig] = getattr(args, "input_db_config", None)
        if not input_db_config:
            raise SystemExit("Database configuration required for database input")
        
        # Try to count rows for progress bar
        try:
            conn = input_db_config.get_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {input_db_config.table}")
            total_rows = cursor.fetchone()[0]
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not count rows: {e}")
            total_rows = 0
        
        doc_stream = stream_texts_from_database(input_db_config, lowercase)
        progress = ProgressBar(total_rows)
        print(f"Processing {total_rows} rows from {input_source_type} database: {input_db_config.database}.{input_db_config.table}")
        
    else:
        # Default: file-based input
        paths = list(iter_text_paths(args.inputs, extensions, args.recursive))
        if not paths:
            raise SystemExit("No input files found.")
        
        doc_stream = stream_texts(paths, lowercase=lowercase)
        progress = ProgressBar(len(paths))
        print(f"Processing {len(paths)} files")

    # Determine output destination type
    output_dest_type = getattr(args, "output_dest_type", "csv")
    output_db_config: Optional[DatabaseConfig] = getattr(args, "output_db_config", None)
    
    if output_dest_type in ("sqlite", "mysql", "postgresql") and output_db_config:
        # Database output - collect results in batches and write to database
        results_batch: List[Tuple[str, List[float]]] = []
        batch_write_size = 100  # Write to DB every 100 results
        total_written = 0
        
        for idx, (doc, meta) in enumerate(
            nlp.pipe(
                doc_stream,
                as_tuples=True,
                batch_size=args.batch_size,
                n_process=args.n_process,
            ),
            start=1,
        ):
            result_values = analyze_doc(doc, morph_dict)
            results_batch.append((meta["text_id"], result_values))
            
            # Write batch to database
            if len(results_batch) >= batch_write_size:
                write_results_to_database(output_db_config, results_batch)
                total_written += len(results_batch)
                results_batch = []
            
            if progress:
                progress.update(idx)
        
        # Write remaining results
        if results_batch:
            write_results_to_database(output_db_config, results_batch)
            total_written += len(results_batch)
        
        if progress:
            progress.close()
        print(f"Wrote {total_written} results to {output_dest_type} database: {output_db_config.table}")
        
    else:
        # CSV output (default)
        with Path(args.output).open("w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["text_id", *COLUMN_NAMES])

            for idx, (doc, meta) in enumerate(
                nlp.pipe(
                    doc_stream,
                    as_tuples=True,
                    batch_size=args.batch_size,
                    n_process=args.n_process,
                ),
                start=1,
            ):
                result_row = [meta["text_id"], *analyze_doc(doc, morph_dict)]
                writer.writerow(result_row)
                if progress:
                    progress.update(idx)
        if progress:
            progress.close()
        print(f"Results written to: {args.output}")


def main() -> None:
    suggested_n_process, suggested_batch, tokens_per_sec = quick_cpu_probe()
    if tokens_per_sec:
        print(
            f"Quick CPU check: ~{tokens_per_sec:,.0f} tokens/sec with a blank spaCy model."
        )
    cores = os.cpu_count() or 1
    print(
        "Suggested settings -> "
        f"batch_size: {suggested_batch}, n_process: {suggested_n_process} "
        f"(detected cores: {cores})"
    )

    args = parse_args()
    if args.batch_size == DEFAULT_BATCH_SIZE:
        args.batch_size = suggested_batch
    if args.n_process == DEFAULT_N_PROCESS:
        args.n_process = suggested_n_process

    # Check if we have any valid input source (files, CSV, or database)
    has_input = (
        args.inputs or 
        getattr(args, 'input_csv_path', '') or 
        getattr(args, 'input_db_config', None)
    )
    
    if args.menu or not has_input:
        args = interactive_menu(args, suggested_n_process, suggested_batch)
    run_tammi(args)


if __name__ == "__main__":
    main()
