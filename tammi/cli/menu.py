"""Interactive menu for TAMMI CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tammi.io.base import AVAILABLE_DRIVERS
from tammi.io.database_io import DatabaseConfig
from tammi.io.csv_io import get_csv_columns
from tammi.cli.progress import check_gpu_available


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
    """Display a Yes/No choice menu."""
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
    """Display suggested integer values as choices."""
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
        try:
            return int(val)
        except ValueError:
            print(f"  Invalid number, using default: {default}")
            return default
    return suggestions[idx]


def _discover_input_folders(base_path: Path = Path(".")) -> List[Path]:
    """Find directories that contain text files."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith((".", "_")):
            txt_files = list(item.glob("*.txt"))[:5]
            if txt_files:
                candidates.append(item)
    return sorted(candidates)


def _discover_csv_files(base_path: Path = Path(".")) -> List[Path]:
    """Find CSV files."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            candidates.append(item)
    return sorted(candidates)


def _discover_json_files(base_path: Path = Path(".")) -> List[Path]:
    """Find JSON files."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_file() and item.suffix.lower() in (".json", ".jsonl"):
            candidates.append(item)
    return sorted(candidates)


def _discover_morpholex_files(base_path: Path = Path(".")) -> List[Path]:
    """Find potential MorphoLex CSV files."""
    candidates = []
    for item in base_path.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            if "morph" in item.name.lower() or "lex" in item.name.lower():
                candidates.append(item)
    return sorted(candidates)


def _discover_spacy_models() -> List[str]:
    """Find installed spaCy models."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "info", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            pipelines = info.get("pipelines", {})
            if pipelines:
                return list(pipelines.keys())
    except Exception:
        pass
            # If spaCy info cannot be retrieved, fall back to default models.
    return ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]


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
    
    elif db_type == "mongodb":
        host = input("  Host [localhost]: ").strip() or "localhost"
        port_str = input("  Port [27017]: ").strip()
        port = int(port_str) if port_str else 27017
        database = input("  Database name: ").strip()
        username = input("  Username (optional): ").strip()
        password = input("  Password (optional): ").strip()
        collection = input("  Collection name [tammi_texts]: ").strip() or "tammi_texts"
        text_col = input("  Text field name [text_content]: ").strip() or "text_content"
        id_col = input("  ID field name [text_id]: ").strip() or "text_id"
        
        return DatabaseConfig(
            db_type="mongodb",
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            table=collection,
            text_column=text_col,
            id_column=id_col,
        )
    
    else:  # MySQL or PostgreSQL
        host = input("  Host [localhost]: ").strip() or "localhost"
        default_port = "3306" if db_type == "mysql" else "5432"
        port_str = input(f"  Port [{default_port}]: ").strip()
        port = int(port_str) if port_str else int(default_port)
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


def interactive_menu(
    base_args: argparse.Namespace,
    suggested_n_process: int,
    suggested_batch: int,
) -> argparse.Namespace:
    """
    Enhanced interactive menu with numbered choices.
    Supports files, CSV, JSON, and database input/output.
    """
    print("\n" + "=" * 60)
    print("  🔬 TAMMI - Tool for Automatic Measurement of Morphological Info")
    print("=" * 60)
    print("  Use number keys to select options. Press Enter for defaults.\n")

    # -------------------------------------------------------------------------
    # 1. INPUT SOURCE TYPE SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("📁 Step 1: Select Input Source")
    
    input_source_options = [
        "Text files (.txt) from folder",
        "CSV file (with text column)",
        "JSON file (with text field)",
    ]
    input_source_types = ["files", "csv", "json"]
    
    if AVAILABLE_DRIVERS.get("sqlite"):
        input_source_options.append("SQLite database")
        input_source_types.append("sqlite")
    if AVAILABLE_DRIVERS.get("mysql"):
        input_source_options.append("MySQL database")
        input_source_types.append("mysql")
    if AVAILABLE_DRIVERS.get("postgresql"):
        input_source_options.append("PostgreSQL database")
        input_source_types.append("postgresql")
    if AVAILABLE_DRIVERS.get("mongodb"):
        input_source_options.append("MongoDB database")
        input_source_types.append("mongodb")
    
    idx, _ = _prompt_choice("Select input source type:", input_source_options, default_index=0)
    input_source_type = input_source_types[idx]
    
    # Variables to hold input config
    inputs: List[str] = []
    input_csv_path = ""
    input_json_path = ""
    input_text_column = "text"
    input_id_column = "id"
    input_db_config: Optional[DatabaseConfig] = None
    ext = ".txt"
    recursive = False
    
    if input_source_type == "files":
        available_folders = _discover_input_folders()
        
        if available_folders:
            folder_options = [f"{f.name}/ ({len(list(f.glob('*.txt')))} .txt files)" for f in available_folders]
            folder_options.append("Enter custom path")
            
            print("\n  Found folders with text files:")
            idx, _ = _prompt_choice("Select input folder:", folder_options, default_index=0)
            
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
        
        recursive = _prompt_yes_no("Search subdirectories recursively?", default=False)
    
    elif input_source_type == "csv":
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
        
        if input_csv_path and Path(input_csv_path).exists():
            columns = get_csv_columns(Path(input_csv_path))
            if columns:
                print(f"\n  Found columns: {', '.join(columns)}")
                
                idx, input_text_column = _prompt_choice(
                    "Select column containing text:",
                    columns,
                    default_index=0,
                    allow_custom=True,
                    custom_label="Enter column name",
                )
                if idx >= 0:
                    input_text_column = columns[idx]
                
                idx, input_id_column = _prompt_choice(
                    "Select column containing text ID:",
                    columns,
                    default_index=0,
                    allow_custom=True,
                    custom_label="Enter column name",
                )
                if idx >= 0:
                    input_id_column = columns[idx]
        
        inputs = [input_csv_path]
    
    elif input_source_type == "json":
        available_jsons = _discover_json_files()
        if available_jsons:
            json_options = [str(f) for f in available_jsons]
            json_options.append("Enter custom JSON path")
            
            idx, _ = _prompt_choice("Select input JSON file:", json_options, default_index=0)
            if idx == len(available_jsons):
                input_json_path = input("\n  Enter JSON file path: ").strip()
            else:
                input_json_path = json_options[idx]
        else:
            input_json_path = input("  Enter JSON file path: ").strip()
        
        input_text_column = input("  Text field name [text_content]: ").strip() or "text_content"
        input_id_column = input("  ID field name [text_id]: ").strip() or "text_id"
        inputs = [input_json_path]
    
    else:
        # Database input
        input_db_config = _configure_database(input_source_type, "input")
        inputs = [f"db:{input_source_type}:{input_db_config.database}"]

    # -------------------------------------------------------------------------
    # 2. OUTPUT DESTINATION SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("💾 Step 2: Select Output Destination")
    
    output_dest_options = ["CSV file", "JSON file"]
    output_dest_types = ["csv", "json"]
    
    if AVAILABLE_DRIVERS.get("sqlite"):
        output_dest_options.append("SQLite database")
        output_dest_types.append("sqlite")
    if AVAILABLE_DRIVERS.get("mysql"):
        output_dest_options.append("MySQL database")
        output_dest_types.append("mysql")
    if AVAILABLE_DRIVERS.get("postgresql"):
        output_dest_options.append("PostgreSQL database")
        output_dest_types.append("postgresql")
    if AVAILABLE_DRIVERS.get("mongodb"):
        output_dest_options.append("MongoDB database")
        output_dest_types.append("mongodb")
    
    idx, _ = _prompt_choice("Select output destination:", output_dest_options, default_index=0)
    output_dest_type = output_dest_types[idx]
    
    output = "morphemes.csv"
    output_db_config: Optional[DatabaseConfig] = None
    
    if output_dest_type == "csv":
        output_suggestions = ["morphemes.csv", "tammi_results.csv", "output/results.csv"]
        
        if inputs and not inputs[0].startswith("db:"):
            input_name = Path(inputs[0]).stem
            output_suggestions.insert(0, f"{input_name}_morphemes.csv")
        
        idx, output = _prompt_choice(
            "Select output filename:",
            output_suggestions,
            default_index=0,
            allow_custom=True,
            custom_label="Enter custom filename/path",
        )
        if idx >= 0:
            output = output_suggestions[idx]
    
    elif output_dest_type == "json":
        output_suggestions = ["morphemes.json", "tammi_results.json", "output/results.json"]
        
        if inputs and not inputs[0].startswith("db:"):
            input_name = Path(inputs[0]).stem
            output_suggestions.insert(0, f"{input_name}_morphemes.json")
        
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

    batch_suggestions = [500, 1000, 1500, 2000, 5000]
    if suggested_batch not in batch_suggestions:
        batch_suggestions.insert(0, suggested_batch)
    batch_suggestions = sorted(set(batch_suggestions))
    
    batch_size = _prompt_int_with_suggestions(
        f"Batch size (suggested: {suggested_batch}):",
        batch_suggestions,
        default=suggested_batch,
    )

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
    elif input_source_type == "json":
        input_summary = f"JSON: {input_json_path} (text: {input_text_column}, id: {input_id_column})"
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
        input_json_path=input_json_path,
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
