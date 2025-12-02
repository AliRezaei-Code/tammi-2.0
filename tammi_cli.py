#!/usr/bin/env python3
"""
Command-line runner for TAMMI (Tool for Automatic Measurement of Morphological Information).

This script rewrites the notebook workflow into a reusable CLI that can process many text
files efficiently. It streams texts through spaCy, loads the MorphoLex CSV once, and writes
results straight to a CSV so very large batches don't require holding everything in memory.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import spacy


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
        description="Run TAMMI on text files and export a CSV of morphological counts.",
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
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Launch an interactive menu to enter options instead of passing arguments.",
    )
    return parser.parse_args()


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
    """
    print("\n" + "=" * 60)
    print("  🔬 TAMMI - Tool for Automatic Measurement of Morphological Info")
    print("=" * 60)
    print("  Use number keys to select options. Press Enter for defaults.\n")

    # -------------------------------------------------------------------------
    # 1. INPUT SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("📁 Step 1: Select Input Files/Folders")
    
    # Discover available folders with text files
    available_folders = _discover_input_folders()
    
    if available_folders:
        folder_options = [f"{f.name}/ ({len(list(f.glob('*.txt')))} .txt files)" for f in available_folders]
        folder_options.append("Enter custom path(s)")
        
        print("\n  Found folders with text files:")
        idx, _ = _prompt_choice(
            "Select input folder:",
            folder_options,
            default_index=0,
        )
        
        if idx == len(available_folders):
            # Custom path
            raw_inputs = input("\n  Enter file/folder paths (comma-separated): ").strip()
            inputs = [p.strip() for p in raw_inputs.split(",") if p.strip()]
        else:
            inputs = [str(available_folders[idx])]
    else:
        # No folders found, ask for manual input
        while True:
            raw_inputs = input("  Enter text files or directories (comma-separated): ").strip()
            inputs = [p.strip() for p in raw_inputs.split(",") if p.strip()]
            if inputs:
                break
            print("  ⚠️  At least one input path is required.")

    # -------------------------------------------------------------------------
    # 2. SPACY MODEL SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("🧠 Step 2: Select spaCy Model")
    
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
    # 3. MORPHOLEX FILE SELECTION
    # -------------------------------------------------------------------------
    _print_menu_header("📖 Step 3: Select MorphoLex Dictionary")
    
    morpholex_files = _discover_morpholex_files()
    if morpholex_files:
        morph_options = [str(f) for f in morpholex_files]
        try:
            default_morph_idx = morph_options.index(base_args.morpholex)
        except ValueError:
            default_morph_idx = 0
        
        idx, morpholex = _prompt_choice(
            "Select MorphoLex CSV file:",
            morph_options,
            default_index=default_morph_idx,
            allow_custom=True,
            custom_label="Enter custom path",
        )
        if idx >= 0:
            morpholex = morph_options[idx]
    else:
        morpholex = _prompt_with_default("  MorphoLex CSV path", base_args.morpholex)

    # -------------------------------------------------------------------------
    # 4. OUTPUT FILE
    # -------------------------------------------------------------------------
    _print_menu_header("💾 Step 4: Output Settings")
    
    output_suggestions = [
        "morphemes.csv",
        "tammi_results.csv", 
        "output/results.csv",
    ]
    # Add input-based suggestion
    if inputs:
        input_name = Path(inputs[0]).stem
        output_suggestions.insert(0, f"{input_name}_morphemes.csv")
    
    idx, output = _prompt_choice(
        "Select output filename:",
        output_suggestions,
        default_index=0,
        allow_custom=True,
        custom_label="Enter custom filename",
    )
    if idx >= 0:
        output = output_suggestions[idx]

    # -------------------------------------------------------------------------
    # 5. FILE EXTENSIONS
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 6. RECURSIVE SEARCH
    # -------------------------------------------------------------------------
    recursive = _prompt_yes_no("Search subdirectories recursively?", default=False)

    # -------------------------------------------------------------------------
    # 7. TEXT CASE
    # -------------------------------------------------------------------------
    _print_menu_header("⚙️  Step 5: Processing Options")
    
    case_options = ["Lowercase text (recommended for TAMMI)", "Keep original case"]
    idx, _ = _prompt_choice("Text case handling:", case_options, default_index=0)
    keep_case = (idx == 1)

    # -------------------------------------------------------------------------
    # 8. GPU vs CPU
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

    # -------------------------------------------------------------------------
    # 9. BATCH SIZE
    # -------------------------------------------------------------------------
    batch_suggestions = [500, 1000, 1500, 2000, 5000]
    if suggested_batch not in batch_suggestions:
        batch_suggestions.insert(0, suggested_batch)
    batch_suggestions = sorted(set(batch_suggestions))
    
    batch_size = _prompt_int_with_suggestions(
        f"Batch size (suggested: {suggested_batch}):",
        batch_suggestions,
        default=suggested_batch,
    )

    # -------------------------------------------------------------------------
    # 10. N_PROCESS (only if not GPU)
    # -------------------------------------------------------------------------
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
    print(f"""
  Input:        {', '.join(inputs)}
  Model:        {model}
  MorphoLex:    {morpholex}
  Output:       {output}
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
        model=model,
        morpholex=morpholex,
        output=output,
        ext=ext,
        recursive=recursive,
        keep_case=keep_case,
        use_gpu=use_gpu,
        batch_size=batch_size,
        n_process=n_process,
        menu=False,
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

    paths = list(iter_text_paths(args.inputs, extensions, args.recursive))
    if not paths:
        raise SystemExit("No input files found.")

    progress: Optional[ProgressBar] = ProgressBar(len(paths))
    with Path(args.output).open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["text_id", *COLUMN_NAMES])

        doc_stream = stream_texts(paths, lowercase=not args.keep_case)
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

    if args.menu or not args.inputs:
        args = interactive_menu(args, suggested_n_process, suggested_batch)
    run_tammi(args)


if __name__ == "__main__":
    main()
