"""Constants and metric calculation helpers for TAMMI."""

from __future__ import annotations

from typing import List


# Column names for TAMMI output
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

# Indices in the MorphoLex CSV that list derivational affixes for a word
DERIVATIONAL_AFFIX_INDICES: tuple[int, ...] = (68, 69, 70, 71, 72, 73, 74)


def safe_divide(a: float, b: float) -> float:
    """Safe division that returns 0 when dividing by zero."""
    return a / b if b else 0.0


def calc_mci(msv: float, between_unique: int, subset_count: int) -> float:
    """Calculate Morphological Complexity Index."""
    return safe_divide(msv + between_unique, subset_count) - 1 if subset_count else 0.0
