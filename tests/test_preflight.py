"""Input failures must be complete and must not create misleading outputs."""
import tempfile
import unittest
from pathlib import Path
from src.data import FILES, build, require_raw_files
import pandas as pd
from src.validation import require_available
from src.model import choose_threshold, metrics


class PreflightTests(unittest.TestCase):
    def test_unknown_target_cannot_become_mature_late_label(self):
        frame = pd.DataFrame({'is_late': [None],
                              'order_estimated_delivery_date': ['2017-01-01'],
                              'order_delivered_customer_date': ['2017-01-02']})
        with self.assertRaisesRegex(ValueError, 'unknown or invalid'):
            require_available(frame, '2018-01-01')

    def test_threshold_maximizes_f1_over_observed_scores(self):
        y, scores = [0, 1, 0, 1, 1, 0], [.1, .2, .2, .4, .8, .9]
        threshold = choose_threshold(y, scores)
        best = max(metrics(y, scores, t)['f1'] for t in set(scores))
        self.assertAlmostEqual(metrics(y, scores, threshold)['f1'], best)

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
