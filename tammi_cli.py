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
import sys
from collections import Counter
from pathlib import Path
from itertools import chain
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
        nargs="+",
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
        default=1000,
        help="spaCy pipe batch size.",
    )
    parser.add_argument(
        "--n-process",
        type=int,
        default=1,
        help="Number of processes for spaCy pipe.",
    )
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Do not lowercase input text (TAMMI originally lowercases).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extensions = tuple(ext.strip().lower() for ext in args.ext.split(",") if ext.strip())

    morph_path = Path(args.morpholex)
    if not morph_path.exists():
        raise SystemExit(f"MorphoLex file not found: {morph_path}")
    morph_dict = load_morph_dict(morph_path)

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


if __name__ == "__main__":
    main()
