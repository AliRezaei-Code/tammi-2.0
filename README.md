# TAMMI

Base code for the Tool for Automatic Measurement of Morphological Information (TAMMI). This does not include code for the graphic user interface.

The user-friendly version of the tool is available at [linguisticanalysistools.org/tammi](https://www.linguisticanalysistools.org/tammi.html)

---

## 🆕 NEW: Command-Line Interface (CLI) Tool

**For users who want to process large batches of text files without Jupyter**, we now offer an optional command-line tool with additional features:

### Quick Start (CLI)

```bash
# Install dependencies
pip install spacy
python -m spacy download en_core_web_sm

# Basic usage - analyze text files
python tammi_cli.py your_texts_folder/ -o results.csv

# Interactive mode (recommended for first-time users)
python tammi_cli.py --menu
```

### What's New in the CLI Tool

| Feature | Description |
|---------|-------------|
| 📁 **Batch Processing** | Process thousands of text files at once |
| 📊 **Multiple Formats** | Input/output as CSV, JSON, or database |
| 🖥️ **Interactive Menu** | Easy-to-use menu for non-technical users |
| ⚡ **GPU Support** | Optional GPU acceleration for large datasets |
| 📈 **Progress Bar** | See real-time progress with time estimates |

### CLI Documentation

- [**User Guide**](docs/user_guide.md) - Step-by-step installation and usage
- [**API Reference**](docs/api_reference.md) - For developers who want to integrate TAMMI
- [**Architecture**](docs/architecture.md) - Technical design documentation

> 💡 **Tip**: The original Jupyter notebook (`Tammi_simp_morpholex_batch_github.ipynb`) is still available and works exactly as before!

---

## Original TAMMI (Jupyter Notebook)

The original TAMMI tool is available as a Jupyter notebook: `Tammi_simp_morpholex_batch_github.ipynb`

### Setup (Notebook)

1. Download or clone this repository
2. Open `Tammi_simp_morpholex_batch_github.ipynb` in Jupyter
3. Ensure `morpho_lex_df_w_log_w_prefsuf_no_head.csv` is in the same folder
4. Run the cells to analyze your texts

---

## About TAMMI 2.0

TAMMI 2.0 was specifically designed to annotate and count morphological features in texts. In developing TAMMI 2.0, we provide automatic calculations for the MorphoLex dataframe provided by Sánchez-Gutiérrez et al. (2017). We also included an automatic calculation of morphological complexity index (MCI) based on inflections as detailed by Brezina and Pallotti (2019). In addition, we calculated an MCI for derivational morphemes and developed new morphological complexity indices based on morphological variety and type-token ratios for both inflectional and derivational morphemes. Lastly, we calculate a number of basic morpheme counts. The indices reported by TAMMI 2.0 are discussed below.

### Basic Morpheme Counts

TAMMI 2.0 includes basic morpheme counts for the number of tokens with inflections and derivational morphemes. The inflections are counted using spaCy (Honnibal & Montani, 2017) by assessing the differences between each token and its lemma. TAMMI also computes the number of words that has prefixes and affixes as well as the number of compound words. In addition, TAMMI 2.0 calculates the total number of prefixes, roots, suffixes, and affixes per text as well as a combination of the number of roots and affixes and a combination of the number of roots, affixes, and inflections. TAMMI 2.0 also computes normed indices by taking the count for each variable and dividing it by 1) the number of content words (i.e., verbs, nouns, adjectives, adverbs) in the text, and by 2) the number of content words with the relevant morpheme.

### Morphological Variety

The inflection morphological variety feature in TAMMI 2.0 is based on a within-subset variety score in which content words from each text are broken into windows of 10 words (plus a window of 1-to-9 for any remaining content words at the end of the text). Inflectional morpheme types (e.g., -s and -ed) for each content word in the window, and null tokens for words without inflections in the window, are counted for each 10-word window and then divided by the total number of windows.

### Morphological Complexity Index (MCI)

TAMMI 2.0 calculates an index for inflectional morphemes based on the MCI reported in Brezina and Pallotti (2019) by using the morphological variety counts above. For inflections, the within-subset variety score is added to the between-subset diversity score. This score is then divided by the number of subsets minus 1. The same approach is followed to produce an MCI for the derivational morphemes.

### Morpheme Type-Token Ratios

TAMMI 2.0 includes indices of type-token ratios for both inflectional and derivational morphemes. For inflectional morphemes, we use the number of unique inflectional morphemes by 10 content word window divided by the length of the window and average the score across the text.

### MorphoLex Variables

TAMMI 2.0 depends on MorphoLex to calculate variables related to frequency/length, family size counts and frequency, and hapax counts. TAMMI 2.0 matches tokens reported in spaCy to the MorphoLex dictionary.

---

## 👥 Contributors

- **TAMMI Original Contributors** - Original implementation and Jupyter notebook
- **Ali Rezaei** (<ali0rezaei0@gmail.com>) - CLI tool and package refactoring:
  - Command-line interface for batch processing
  - JSON/JSONL input/output support  
  - MongoDB database support
  - Interactive menu for ease of use
  - Comprehensive documentation

## 📄 License

MIT License

## 🔗 References

- Sánchez-Gutiérrez, C. H., Mailhot, H., Deacon, S. H., & Wilson, M. A. (2017). MorphoLex: A derivational morphological database for 70,000 English words. *Behavior Research Methods*.
- Brezina, V., & Pallotti, G. (2019). Morphological complexity in written L2 texts. *Second Language Research*.
- Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.
