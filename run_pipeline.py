"""Run from any working directory: python /path/to/run_pipeline.py."""
from pathlib import Path
import argparse, json, time
from src.data import build
from src.model import train_models
from src.analysis import analyze
from src.diagnostics import diagnostics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-training', action='store_true', help='Only rebuild audits and descriptive figures')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    start = time.time()
    d, e, t = build(root)
    print(f'All orders: {len(d)}; eligible: {len(e)}; late: {e.is_late.sum()}', flush=True)
    result = None if args.skip_training else train_models(e, root)
    analyze(d, e, root, result)
    if result is not None:
        diagnostics(root)
    print(f"Completed in {time.time() - start:.1f}s. Outputs: {root / 'outputs'}", flush=True)
if __name__ == '__main__':
    main()
