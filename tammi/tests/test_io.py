"""Tests for I/O module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tammi.io.base import ReaderFactory, WriterFactory
from tammi.io.csv_io import CSVReader, CSVWriter
from tammi.io.json_io import JSONReader, JSONWriter, JSONLReader, JSONLWriter
from tammi.io.file_io import TextFileReader


class TestCSVIO(unittest.TestCase):
    """Test CSV input/output."""
    
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = Path(self.temp_dir) / "test.csv"
        
        # Create test CSV
        with self.test_csv.open("w") as f:
            f.write("text_id,text_content,extra\n")
            f.write("doc1,Hello world,ignored\n")
            f.write("doc2,Testing TAMMI,also ignored\n")
            f.write("doc3,Morphological analysis,more data\n")
    
    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_reader_stream(self) -> None:
        reader = CSVReader(self.test_csv, text_column="text_content", id_column="text_id")
        records = list(reader.stream(lowercase=False))
        
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0][0], "Hello world")
        self.assertEqual(records[0][1]["text_id"], "doc1")
    
    def test_csv_reader_count(self) -> None:
        reader = CSVReader(self.test_csv, text_column="text_content", id_column="text_id")
        self.assertEqual(reader.count(), 3)
    
    def test_csv_reader_lowercase(self) -> None:
        reader = CSVReader(self.test_csv, text_column="text_content", id_column="text_id")
        records = list(reader.stream(lowercase=True))
        
        self.assertEqual(records[0][0], "hello world")
    
    def test_csv_writer(self) -> None:
        output_path = Path(self.temp_dir) / "output.csv"
        columns = ["metric1", "metric2"]
        
        with CSVWriter(output_path, columns) as writer:
            writer.write_header(columns)
            writer.write_record("doc1", [0.5, 0.8])
            writer.write_record("doc2", [0.3, 0.9])
        
        # Read back and verify
        with output_path.open() as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 3)  # header + 2 records
        self.assertIn("text_id", lines[0])
        self.assertIn("metric1", lines[0])


class TestJSONIO(unittest.TestCase):
    """Test JSON input/output."""
    
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test JSON (array format)
        self.test_json_array = Path(self.temp_dir) / "test_array.json"
        with self.test_json_array.open("w") as f:
            json.dump([
                {"text_id": "doc1", "text_content": "Hello world"},
                {"text_id": "doc2", "text_content": "Testing TAMMI"},
            ], f)
        
        # Create test JSON (object with records key)
        self.test_json_object = Path(self.temp_dir) / "test_object.json"
        with self.test_json_object.open("w") as f:
            json.dump({
                "records": [
                    {"text_id": "doc1", "text_content": "Hello world"},
                    {"text_id": "doc2", "text_content": "Testing TAMMI"},
                ]
            }, f)
        
        # Create test JSONL
        self.test_jsonl = Path(self.temp_dir) / "test.jsonl"
        with self.test_jsonl.open("w") as f:
            f.write(json.dumps({"text_id": "doc1", "text_content": "Hello world"}) + "\n")
            f.write(json.dumps({"text_id": "doc2", "text_content": "Testing TAMMI"}) + "\n")
    
    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_reader_array_format(self) -> None:
        reader = JSONReader(self.test_json_array, text_column="text_content", id_column="text_id")
        records = list(reader.stream(lowercase=False))
        
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], "Hello world")
        self.assertEqual(records[0][1]["text_id"], "doc1")
    
    def test_json_reader_object_format(self) -> None:
        reader = JSONReader(self.test_json_object, text_column="text_content", id_column="text_id")
        records = list(reader.stream(lowercase=False))
        
        self.assertEqual(len(records), 2)
    
    def test_json_reader_count(self) -> None:
        reader = JSONReader(self.test_json_array, text_column="text_content", id_column="text_id")
        self.assertEqual(reader.count(), 2)
    
    def test_jsonl_reader(self) -> None:
        reader = JSONLReader(self.test_jsonl, text_column="text_content", id_column="text_id")
        records = list(reader.stream(lowercase=False))
        
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], "Hello world")
    
    def test_json_writer(self) -> None:
        output_path = Path(self.temp_dir) / "output.json"
        columns = ["metric1", "metric2"]
        
        with JSONWriter(output_path, columns) as writer:
            writer.write_record("doc1", [0.5, 0.8])
            writer.write_record("doc2", [0.3, 0.9])
        
        # Read back and verify
        with output_path.open() as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("records", data)
        self.assertEqual(len(data["records"]), 2)
        self.assertEqual(data["records"][0]["text_id"], "doc1")
        self.assertEqual(data["records"][0]["metric1"], 0.5)
    
    def test_jsonl_writer(self) -> None:
        output_path = Path(self.temp_dir) / "output.jsonl"
        columns = ["metric1", "metric2"]
        
        with JSONLWriter(output_path, columns) as writer:
            writer.write_record("doc1", [0.5, 0.8])
            writer.write_record("doc2", [0.3, 0.9])
        
        # Read back and verify
        with output_path.open() as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        record1 = json.loads(lines[0])
        self.assertEqual(record1["text_id"], "doc1")
        self.assertEqual(record1["metric1"], 0.5)


class TestFileIO(unittest.TestCase):
    """Test text file input."""
    
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test text files
        (Path(self.temp_dir) / "doc1.txt").write_text("Hello world")
        (Path(self.temp_dir) / "doc2.txt").write_text("Testing TAMMI")
        (Path(self.temp_dir) / "ignored.md").write_text("Markdown file")
    
    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_reader_stream(self) -> None:
        reader = TextFileReader([self.temp_dir], extensions=(".txt",))
        records = list(reader.stream(lowercase=False))
        
        self.assertEqual(len(records), 2)
    
    def test_file_reader_count(self) -> None:
        reader = TextFileReader([self.temp_dir], extensions=(".txt",))
        self.assertEqual(reader.count(), 2)
    
    def test_file_reader_lowercase(self) -> None:
        reader = TextFileReader([self.temp_dir], extensions=(".txt",))
        records = list(reader.stream(lowercase=True))
        
        texts = [r[0] for r in records]
        self.assertTrue(all(t == t.lower() for t in texts))


class TestReaderFactory(unittest.TestCase):
    """Test reader factory."""
    
    def test_factory_available_types(self) -> None:
        types = ReaderFactory.available_types()
        self.assertIn("csv", types)
        self.assertIn("json", types)
        self.assertIn("files", types)
    
    def test_factory_invalid_type(self) -> None:
        with self.assertRaises(ValueError):
            ReaderFactory.create("invalid_type")


class TestWriterFactory(unittest.TestCase):
    """Test writer factory."""
    
    def test_factory_available_types(self) -> None:
        types = WriterFactory.available_types()
        self.assertIn("csv", types)
        self.assertIn("json", types)


if __name__ == "__main__":
    unittest.main()
