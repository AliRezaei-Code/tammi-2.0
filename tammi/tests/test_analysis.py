"""Tests for analysis module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tammi.analysis.morpholex import MorphoLexDict
from tammi.analysis.metrics import COLUMN_NAMES, safe_divide, calc_mci


class TestMetrics(unittest.TestCase):
    """Test metric calculation helpers."""
    
    def test_safe_divide_normal(self) -> None:
        self.assertEqual(safe_divide(10, 2), 5.0)
    
    def test_safe_divide_by_zero(self) -> None:
        self.assertEqual(safe_divide(10, 0), 0.0)
    
    def test_calc_mci(self) -> None:
        # Basic MCI calculation
        result = calc_mci(5.0, 3, 4)
        self.assertIsInstance(result, float)
    
    def test_calc_mci_zero_subset(self) -> None:
        result = calc_mci(5.0, 3, 0)
        self.assertEqual(result, 0.0)
    
    def test_column_names_count(self) -> None:
        # Should have 42 columns
        self.assertEqual(len(COLUMN_NAMES), 42)


class TestMorphoLexDict(unittest.TestCase):
    """Test MorphoLex dictionary."""
    
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal test MorphoLex file
        self.test_morpholex = Path(self.temp_dir) / "test_morpholex.csv"
        with self.test_morpholex.open("w") as f:
            # word, followed by 75 values
            values = ",".join(["0.5"] * 75)
            f.write(f"test,{values}\n")
            f.write(f"hello,{values}\n")
            f.write(f"world,{values}\n")
    
    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_and_get(self) -> None:
        morph = MorphoLexDict(self.test_morpholex)
        
        self.assertIn("test", morph)
        self.assertIn("hello", morph)
        self.assertNotIn("notfound", morph)
    
    def test_get_values(self) -> None:
        morph = MorphoLexDict(self.test_morpholex)
        
        values = morph.get("test")
        self.assertIsNotNone(values)
        self.assertEqual(len(values), 75)
    
    def test_len(self) -> None:
        morph = MorphoLexDict(self.test_morpholex)
        self.assertEqual(len(morph), 3)
    
    def test_getitem(self) -> None:
        morph = MorphoLexDict(self.test_morpholex)
        values = morph["test"]
        self.assertEqual(len(values), 75)
    
    def test_getitem_missing(self) -> None:
        morph = MorphoLexDict(self.test_morpholex)
        with self.assertRaises(KeyError):
            _ = morph["notfound"]


if __name__ == "__main__":
    unittest.main()
