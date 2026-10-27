"""Input failures must be complete and must not create misleading outputs."""
import tempfile
import unittest
from pathlib import Path
from src.data import FILES, build, require_raw_files


class PreflightTests(unittest.TestCase):
    def test_lists_every_missing_input_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError) as caught:
                build(root)
            for filename in FILES.values():
                self.assertIn(filename, str(caught.exception))
            self.assertFalse((root / 'outputs').exists())
            self.assertFalse((root / 'data/processed').exists())

    def test_reports_only_absent_files(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            present = next(iter(FILES.values()))
            (raw / present).write_text('placeholder for presence check only')
            with self.assertRaises(FileNotFoundError) as caught:
                require_raw_files(raw)
            self.assertNotIn(present, str(caught.exception))
            self.assertIn('olist_customers_dataset.csv', str(caught.exception))
