"""Post-selection diagnostics. No outputs from this module alter fitting or thresholds."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve


def diagnostics(root):
    root = Path(root)
    tab, figures = root / 'outputs/tables', root / 'outputs/figures'
    def save(fig, name):
        fig.tight_layout()
        temp = figures / f'{name}.tmp.png'
        fig.savefig(temp, dpi=180, bbox_inches='tight')
        temp.replace(figures / f'{name}.png')
        plt.close(fig)
    results = pd.read_csv(tab / 'test_metrics.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, row in zip(axes.flat, results.to_dict('records')):
        matrix = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]])
        sns.heatmap(matrix, annot=True, fmt='d', cbar=False, cmap='Blues', ax=ax,
                    xticklabels=['On time', 'Late'], yticklabels=['On time', 'Late'])
        ax.set(title=row['model'], xlabel='Predicted', ylabel='Actual')
    for ax in axes.flat[len(results):]:
        ax.axis('off')
    save(fig, '11_confusion_matrices')
    cv = pd.read_csv(tab / 'cv_summary.csv').sort_values('mean_ap')
    fig, ax = plt.subplots(figsize=(9, 5.1))
    ax.barh(cv.model, cv.mean_ap, xerr=cv.sd_ap, color='#176B87', capsize=3)
    ax.set(title='Three temporal folds: mean Average Precision',
           xlabel='Mean AP with standard deviation across folds')
    save(fig, '14_temporal_cv')
    import json
    selected = json.loads((tab / 'selection.json').read_text())['winner']
    scores = pd.read_csv(tab / 'all_test_scores.csv')
    scores = scores.loc[scores.model.eq(selected)]
    fig, ax = plt.subplots(figsize=(7.5, 5.1))
    curve_values = []
    for column, label in [('raw_score', 'Raw score'), ('score', 'Sigmoid calibrated')]:
        fraction, mean = calibration_curve(scores.is_late, scores[column], n_bins=8, strategy='quantile')
        ax.plot(mean, fraction, marker='o', label=label)
        curve_values.extend(mean.tolist() + fraction.tolist())
    ax.set(title='Historical final-period reliability diagnostic',
           xlabel='Mean predicted score in quantile bin', ylabel='Observed late fraction')
    limit = min(1., max([.10] + curve_values) * 1.2)
    ax.plot([0, limit], [0, limit], '--', color='gray', label='Ideal calibration')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.legend()
    save(fig, '15_calibration')
    manifest = pd.read_csv(tab / 'figure_manifest.csv')
    manifest = manifest.loc[~manifest.figure.isin(['14_temporal_cv', '15_calibration'])]
    additions = pd.DataFrame([
        {'figure':'14_temporal_cv','kind':'bar','population':'Temporal CV folds, aggregate of 3 evaluations','n':len(cv)},
        {'figure':'15_calibration','kind':'line','population':'Historical final chronological period','n':len(scores)},
    ])
    pd.concat([manifest, additions], ignore_index=True).to_csv(tab / 'figure_manifest.csv', index=False)
