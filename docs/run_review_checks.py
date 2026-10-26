"""Reproduce review comparisons from frozen outputs; never fit or retune."""
from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tables'
def main():
    selected = json.loads((OUT/'selection.json').read_text())['winner']
    current = pd.read_csv(OUT/'test_metrics.csv').set_index('model').loc[selected]
    legacy = pd.read_csv(ROOT/'outputs/legacy_v1/test_metrics.csv')
    old = legacy.loc[legacy.model.ne('Dummy')].sort_values('f1',ascending=False).iloc[0]
    rows = [{'metric':key,'v1':float(old[key]),'v2':float(current[key])} for key in ['precision','recall','f1','average_precision','fp','fn']]
    pd.DataFrame(rows).to_csv(OUT/'review_comparison.csv',index=False)
    summary = {'winner':selected,'false_alert_share':float(current.fp/(current.fp+current.tp)), 'false_alerts_per_true_alert':float(current.fp/current.tp), 'test_is_new_untouched_holdout':False,'retuned_by_review':False,'caveat':'Historical comparison; lower false alerts accompanied lower recall. See experiment history.'}
    (OUT/'review_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__ == '__main__':
    main()
