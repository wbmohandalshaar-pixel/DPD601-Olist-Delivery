"""Meaningful regression checks for target, joins, leakage and temporal boundaries."""
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.data import target, FEATURES, FORBIDDEN, FILES, KEYS
from src.model import temporal_split, choose_threshold, metrics, preprocessing
ROOT = Path(__file__).resolve().parents[1]
RAW_INPUTS_AVAILABLE = all((ROOT / 'data/raw' / filename).is_file() for filename in FILES.values())
RAW_INPUTS_NOTE = ('The seven original Kaggle CSVs are intentionally not included in this '
                   'university submission. Add them to data/raw to run raw-data checks.')

class TargetTests(unittest.TestCase):

    def test_same_calendar_day_is_on_time(self):
        self.assertEqual(target(pd.Series(['2018-03-01 23:59:59']), pd.Series(['2018-03-01 00:00:00'])).iloc[0], 0)

    def test_next_day_is_late(self):
        self.assertEqual(target(pd.Series(['2018-03-02 00:00:00']), pd.Series(['2018-03-01'])).iloc[0], 1)

    def test_earlier_day_is_on_time(self):
        self.assertEqual(target(pd.Series(['2018-02-28']), pd.Series(['2018-03-01'])).iloc[0], 0)

    def test_missing_actual_unknown(self):
        self.assertTrue(pd.isna(target(pd.Series([None]), pd.Series(['2018-03-01'])).iloc[0]))

    def test_missing_estimate_unknown(self):
        self.assertTrue(pd.isna(target(pd.Series(['2018-03-01']), pd.Series([None])).iloc[0]))

    def test_invalid_date_unknown(self):
        self.assertTrue(pd.isna(target(pd.Series(['bad']), pd.Series(['2018-03-01'])).iloc[0]))

class ArtifactTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.all = pd.read_csv(ROOT / 'data/processed/all_orders.csv', dtype={'order_id': 'string'})
        cls.d = pd.read_csv(ROOT / 'data/processed/model_orders.csv', parse_dates=['order_approved_at', 'order_delivered_customer_date'])

    @unittest.skipUnless(RAW_INPUTS_AVAILABLE, RAW_INPUTS_NOTE)
    def test_primary_keys(self):
        for n, f in FILES.items():
            d = pd.read_csv(ROOT / 'data/raw' / f)
            self.assertFalse(d.duplicated(KEYS[n]).any(), n)
            self.assertFalse(d[KEYS[n]].isna().any().any(), n)

    def test_one_row_per_order(self):
        self.assertEqual(len(self.all), 99441)
        self.assertTrue(self.all.order_id.is_unique)
        self.assertTrue(self.d.order_id.is_unique)

    def test_join_counts(self):
        a = pd.read_csv(ROOT / 'outputs/tables/join_audit.csv')
        self.assertTrue(a.before.eq(a.after).all())

    @unittest.skipUnless(RAW_INPUTS_AVAILABLE, RAW_INPUTS_NOTE)
    def test_aggregate_conservation(self):
        it = pd.read_csv(ROOT / 'data/raw' / FILES['items'])
        pay = pd.read_csv(ROOT / 'data/raw' / FILES['payments'])
        for c, s in [('total_price', it.price.sum()), ('total_freight', it.freight_value.sum()), ('payment_total', pay.payment_value.sum())]:
            self.assertTrue(np.isclose(self.all[c].sum(), s), c)

    def test_target_matches_calendar_definition(self):
        expected = target(self.d.order_delivered_customer_date, pd.to_datetime(self.d.order_estimated_delivery_date))
        np.testing.assert_array_equal(self.d.is_late, expected)

    def test_only_eligible_delivered(self):
        self.assertTrue(self.d.order_status.eq('delivered').all())
        self.assertTrue(self.d.eligibility_reason.eq('eligible').all())
        self.assertFalse(self.d.is_late.isna().any())

    def test_no_forbidden_features(self):
        self.assertFalse(set(FEATURES) & FORBIDDEN)
        self.assertTrue(set(FEATURES) <= set(self.d.columns))

    def test_temporal_disjoint_and_mature(self):
        a, b, c = temporal_split(self.d)
        for x, y in [(a, b), (b, c)]:
            self.assertFalse(set(x.order_id) & set(y.order_id))
            self.assertLess(x.order_approved_at.max(), y.order_approved_at.min())
            self.assertLess(__import__('src.validation', fromlist=['label_available_at']).label_available_at(x).max(), y.order_approved_at.min())

    def test_exclusion_reconciliation(self):
        ex = pd.read_csv(ROOT / 'outputs/tables/exclusions.csv')
        self.assertEqual(ex.orders.sum(), len(self.all))
        self.assertEqual(ex.loc[ex.reason.eq('eligible'), 'orders'].iloc[0], len(self.d))

    def test_temporal_invariants(self):
        self.assertTrue(self.d.planned_days.ge(0).all())
        self.assertTrue(self.d.approval_wait_hours.ge(0).all())

    @unittest.skipUnless(RAW_INPUTS_AVAILABLE, RAW_INPUTS_NOTE)
    def test_postal_prefixes_preserved(self):
        from src.data import load
        t, _ = load(ROOT / 'data/raw')
        self.assertEqual(str(t['customers'].customer_zip_code_prefix.dtype), 'string')
        self.assertTrue(t['customers'].customer_zip_code_prefix.str.len().eq(5).all())


    def test_saved_selection_and_metrics_consistent(self):
        path = ROOT / 'outputs/tables/test_metrics.csv'
        if not path.exists():
            self.skipTest('Run full pipeline first')
        s = json.loads((ROOT / 'outputs/tables/selection.json').read_text())
        r = pd.read_csv(path)
        pred = pd.read_csv(ROOT / 'outputs/tables/test_predictions.csv')
        m = metrics(pred.is_late, pred.probability, s['threshold'])
        row = r.set_index('model').loc[s['winner']]
        for k in ['tp', 'tn', 'fp', 'fn', 'f1', 'average_precision']:
            self.assertAlmostEqual(m[k], row[k], places=10)


    @unittest.skipUnless(RAW_INPUTS_AVAILABLE, RAW_INPUTS_NOTE)
    def test_raw_hashes_match_manifest(self):
        import hashlib
        manifest = json.loads((ROOT / 'outputs/tables/raw_sha256.json').read_text())
        for name, digest in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT / 'data/raw' / name).read_bytes()).hexdigest(), digest)


    def test_saved_pipeline_reproduces_test_scores(self):
        import joblib
        model = ROOT / 'outputs/models/selected_model.joblib'
        if not model.exists():
            self.skipTest('Run full pipeline first')
        bundle = joblib.load(model)
        saved = pd.read_csv(ROOT / 'outputs/tables/test_predictions.csv')
        ids = saved.order_id.iloc[:200]
        sample = self.d.set_index('order_id').loc[ids]
        from src.model import calibrated_scores
        scores = calibrated_scores(bundle['pipeline'], bundle.get('calibrator'), sample[bundle['features']])
        np.testing.assert_allclose(scores, saved.probability.iloc[:200], rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(RAW_INPUTS_AVAILABLE, RAW_INPUTS_NOTE)
    def test_fresh_checkout_recreates_processed_directory(self):
        import tempfile, shutil
        from src.data import build
        with tempfile.TemporaryDirectory() as folder:
            fresh = Path(folder)
            shutil.copytree(ROOT / 'data/raw', fresh / 'data/raw')
            self.assertFalse((fresh / 'data/processed').exists())
            all_orders, eligible, _ = build(fresh)
            self.assertEqual(len(all_orders), 99441)
            self.assertEqual(len(eligible), len(self.d))
            self.assertTrue((fresh / 'data/processed/all_orders.csv').is_file())
            self.assertTrue((fresh / 'outputs/tables/join_audit.csv').is_file())

    def test_cohort_membership_does_not_depend_on_arrival_speed(self):
        altered = self.d.copy()
        late = altered.is_late.eq(1)
        altered.loc[late, 'order_delivered_customer_date'] += pd.Timedelta(days=250)
        before = temporal_split(self.d)
        after = temporal_split(altered)
        for first, second in zip(before, after):
            self.assertEqual(set(first.order_id), set(second.order_id))



    def test_validation_blocks_are_disjoint(self):
        cal = pd.read_csv(ROOT / 'outputs/tables/calibration_membership.csv')
        threshold = pd.read_csv(ROOT / 'outputs/tables/threshold_membership.csv')
        whole = pd.read_csv(ROOT / 'outputs/tables/validation_membership.csv')
        self.assertFalse(set(cal.order_id) & set(threshold.order_id))
        self.assertEqual(set(cal.order_id) | set(threshold.order_id), set(whole.order_id))

    def test_all_cv_trials_use_three_folds(self):
        cv = pd.read_csv(ROOT / 'outputs/tables/temporal_cv_trials.csv')
        from src.model import model_specs
        self.assertEqual(set(cv.model), {name for name, _, _ in model_specs()})
        self.assertFalse(cv.duplicated(['model', 'fold']).any())
        self.assertEqual(len(cv), 3 * len(model_specs()))
        self.assertTrue(cv.groupby('model').fold.nunique().eq(3).all())
        summary = pd.read_csv(ROOT / 'outputs/tables/cv_summary.csv')
        expected = cv.groupby('model').average_precision.mean()
        for row in summary.itertuples():
            self.assertAlmostEqual(row.mean_ap, expected[row.model], places=12)

    def test_selected_model_comes_from_cv_only(self):
        summary = pd.read_csv(ROOT / 'outputs/tables/cv_summary.csv')
        selection = json.loads((ROOT / 'outputs/tables/selection.json').read_text())
        self.assertEqual(selection['winner'], summary.iloc[0].model)
        self.assertIn('mean temporal CV', selection['selection'])

    def test_calibration_ranking_is_monotone(self):
        import joblib
        from src.model import score_logit
        bundle = joblib.load(ROOT / 'outputs/models/selected_model.joblib')
        cal = bundle.get('calibrator')
        if cal is not None:
            self.assertGreater(cal.coef_[0, 0], 0)
            self.assertTrue(np.diff(cal.predict_proba(score_logit([0.01, 0.1, 0.5, 0.9]))[:, 1]).min() > 0)


class LogicTests(unittest.TestCase):

    def test_preprocessor_unseen_category_and_train_median(self):
        from src.data import NUMERIC, CATEGORICAL
        train = pd.DataFrame({**{k: [1.0, 3.0] for k in NUMERIC}, **{k: ['a', 'a'] for k in CATEGORICAL}})
        test = pd.DataFrame({**{k: [np.nan] for k in NUMERIC}, **{k: ['unseen'] for k in CATEGORICAL}})
        p = preprocessing()
        p.fit(train)
        trans = p.transform(test)
        self.assertEqual(trans.shape[0], 1)
        np.testing.assert_array_equal(p.named_transformers_['num'].named_steps['impute'].statistics_, np.repeat(2.0, len(NUMERIC)))


    def test_feature_dictionary_exactly_matches_inputs(self):
        dictionary = pd.read_csv(ROOT / 'docs/feature_dictionary.csv')
        self.assertEqual(set(dictionary.feature), set(FEATURES))
        self.assertEqual(len(dictionary), len(FEATURES))


    def test_many_to_many_fixture_demonstrates_inflation(self):
        items = pd.DataFrame({'order_id': ['a'] * 3, 'price': [10.0, 20.0, 30.0]})
        payments = pd.DataFrame({'order_id': ['a'] * 2, 'payment_value': [25.0, 35.0]})
        bad = items.merge(payments, on='order_id')
        self.assertEqual(len(bad), 6)
        self.assertEqual(bad.price.sum(), 120.0)
        good = items.groupby('order_id', as_index=False).price.sum().merge(payments.groupby('order_id', as_index=False).payment_value.sum(), on='order_id', validate='one_to_one')
        self.assertEqual(len(good), 1)
        self.assertEqual(good.price.sum(), 60.0)
        self.assertEqual(good.payment_value.sum(), 60.0)


    def test_late_label_known_at_first_overdue_midnight(self):
        from src.validation import label_available_at
        sample = pd.DataFrame({'is_late': [1, 0], 'order_estimated_delivery_date': ['2018-01-10', '2018-01-10'], 'order_delivered_customer_date': ['2018-03-01 12:00:00', '2018-01-10 23:59:00']})
        actual = label_available_at(sample)
        self.assertEqual(actual.iloc[0], pd.Timestamp('2018-01-11'))
        self.assertEqual(actual.iloc[1], pd.Timestamp('2018-01-10 23:59:00'))


    def test_unknown_labels_stop_whole_cohort(self):
        from src.validation import require_available
        sample = pd.DataFrame({'is_late': [0], 'order_estimated_delivery_date': ['2018-01-10'], 'order_delivered_customer_date': ['2018-01-10 12:00:00']})
        with self.assertRaises(ValueError):
            require_available(sample, '2018-01-05')


    def test_log_cyclic_transform_preserves_missingness(self):
        from src.data import NUMERIC
        from src.features import log_cyclic_features
        x = pd.DataFrame({k: [1.0, np.nan] for k in NUMERIC})
        x['approval_hour'] = [23.0, 0.0]
        changed = log_cyclic_features(x)
        self.assertTrue(pd.isna(changed.total_price.iloc[1]))
        self.assertAlmostEqual(changed.total_price.iloc[0], np.log(2))
        self.assertNotIn('approval_hour', changed)
        self.assertIn('approval_hour_sin', changed)


    def test_threshold_uses_scores(self):
        self.assertGreaterEqual(choose_threshold([0, 1, 1], [0.1, 0.4, 0.8]), 0.1)


if __name__ == '__main__':
    unittest.main()
