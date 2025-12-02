"""Main TAMMI analyzer - performs morphological analysis on text documents."""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

from tammi.analysis.metrics import (
    COLUMN_NAMES,
    DERIVATIONAL_AFFIX_INDICES,
    safe_divide,
    calc_mci,
)
from tammi.analysis.morpholex import MorphoLexDict

if TYPE_CHECKING:


def list_windows(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield successive windows of the given size from a sequence."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class TAMMIAnalyzer:
    """
    TAMMI analyzer for morphological complexity analysis.
    
    Uses spaCy for tokenization and lemmatization, and the MorphoLex
    dictionary for morphological information.
    
    Example usage:
        analyzer = TAMMIAnalyzer(morpholex_path="morpho_lex.csv")
        results = analyzer.analyze_text("The quick brown fox jumps over the lazy dog.")
    """
    
    def __init__(
        self,
        morpholex: MorphoLexDict | str | None = None,
        spacy_model: str = "en_core_web_sm",
        use_gpu: bool = False,
    ) -> None:
        """
        Initialize the TAMMI analyzer.
        
        Args:
            morpholex: MorphoLexDict instance or path to MorphoLex CSV
            spacy_model: Name of spaCy model to use
            use_gpu: Whether to enable GPU acceleration
        """
        self._nlp: Optional[Any] = None
        self._model_name = spacy_model
        self._use_gpu = use_gpu
        
        # Load MorphoLex
        if isinstance(morpholex, MorphoLexDict):
            self._morpholex = morpholex
        elif morpholex is not None:
            self._morpholex = MorphoLexDict(morpholex)
        else:
            self._morpholex = MorphoLexDict()
    
    @property
    def nlp(self) -> Any:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            
            if self._use_gpu:
                try:
                    spacy.prefer_gpu()  # type: ignore[attr-defined]
                except Exception:
                    pass  # GPU not available, continue with CPU
            
            self._nlp = spacy.load(self._model_name)
        return self._nlp
    
    @property
    def morpholex(self) -> MorphoLexDict:
        """Return the MorphoLex dictionary."""
        return self._morpholex
    
    @property
    def column_names(self) -> List[str]:
        """Return the column names for output metrics."""
        return COLUMN_NAMES.copy()
    
    def load_morpholex(self, path: str) -> None:
        """Load or reload the MorphoLex dictionary."""
        self._morpholex.load(path)
    
    def analyze_doc(self, doc: Any) -> List[float]:
        """
        Analyze a spaCy Doc and return morphological metrics.
        
        Args:
            doc: A spaCy Doc object
            
        Returns:
            List of float values corresponding to COLUMN_NAMES
        """
        morph_dict = self._morpholex.data
        
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

        def cal_fixes(fix_var: List[Any]) -> None:
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
    
    def analyze_text(self, text: str, lowercase: bool = True) -> List[float]:
        """
        Analyze a text string and return morphological metrics.
        
        Args:
            text: The text to analyze
            lowercase: Whether to lowercase the text first
            
        Returns:
            List of float values corresponding to COLUMN_NAMES
        """
        if lowercase:
            text = text.lower()
        doc = self.nlp(text)
        return self.analyze_doc(doc)
    
    def analyze_texts(
        self,
        texts: Iterator[Tuple[str, Dict[str, str]]],
        batch_size: int = 1000,
        n_process: int = 1,
    ) -> Iterator[Tuple[str, List[float]]]:
        """
        Analyze multiple texts using spaCy's pipe for efficiency.
        
        Args:
            texts: Iterator of (text, metadata) tuples where metadata has 'text_id'
            batch_size: Batch size for spaCy pipe
            n_process: Number of processes for parallel processing
            
        Yields:
            Tuples of (text_id, metrics)
        """
        for doc, meta in self.nlp.pipe(
            texts,
            as_tuples=True,
            batch_size=batch_size,
            n_process=n_process,
        ):
            yield meta["text_id"], self.analyze_doc(doc)
