"""Batch scores from engineered, approval-time order features; never load untrusted joblib."""
import argparse
from pathlib import Path
import pandas as pd
import joblib
from src.model import calibrated_scores

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='CSV containing engineered FEATURES plus optional order_id')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    bundle = joblib.load(root / 'outputs/models/selected_model.joblib')
    d = pd.read_csv(args.input, dtype={'order_id': 'string'})
    missing = set(bundle['features']) - set(d)
    if missing:
        raise ValueError(f'Missing engineered features: {sorted(missing)}')
    score = calibrated_scores(bundle['pipeline'], bundle.get('calibrator'), d[bundle['features']])
    out = pd.DataFrame({'late_score': score, 'predicted_is_late': (score >= bundle['threshold']).astype(int)})
    if 'order_id' in d:
        out.insert(0, 'order_id', d.order_id)
    out.to_csv(args.output, index=False)
if __name__ == '__main__':
    main()
