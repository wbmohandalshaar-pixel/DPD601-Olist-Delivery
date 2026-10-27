"""Temporal model comparison, independent calibration and threshold blocks, fixed final evaluation."""
from pathlib import Path
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from .features import log_cyclic_features

from .data import CATEGORICAL, FEATURES, NUMERIC
from .validation import final_split, temporal_folds, require_available

SEED = 601


def preprocessing(feature_mode="raw"):
    steps = []
    if feature_mode == "log_cyclic":
        steps.append(("deterministic_features", FunctionTransformer(log_cyclic_features)))
    numeric = Pipeline(steps + [
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([('num', numeric, NUMERIC), ('cat', categorical, CATEGORICAL)])


def metrics(y, scores, threshold):
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'average_precision': float(average_precision_score(y, scores)),
        'roc_auc': float(roc_auc_score(y, scores)),
        'brier': float(brier_score_loss(y, scores)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'alert_rate': float(np.mean(predicted)),
    }


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(thresholds), where=denominator > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def model_specs():
    """Finite candidate set fixed in code before the revised evaluation."""
    result = [('Dummy', 'Dummy', {})]
    for c in [.1, 1.]:
        for weight in [None, 'balanced']:
            name = f'Logistic_C{c:g}_{weight or "plain"}'
            result.append((name, 'Logistic', {'C': c, 'class_weight': weight}))
    for c in [.1, 1.]:
        for weight in [None, 'balanced']:
            result.append((f'LogisticLog_C{c:g}_{weight or "plain"}', 'Logistic',
                           {'C': c, 'class_weight': weight, 'feature_mode': 'log_cyclic'}))
    # Finite extension registered before this run: vary regularization only.
    for c in [.03, .3, 3.]:
        result.append((f'LogisticLog_C{c:g}_plain', 'Logistic',
                       {'C': c, 'class_weight': None, 'feature_mode': 'log_cyclic'}))
    for leaf in [10, 30]:
        result.append((f'RF_leaf{leaf}', 'RandomForest', {'min_samples_leaf': leaf}))
    for leaves in [15, 31]:
        result.append((f'HGB_leaf{leaves}', 'HistGradientBoosting', {'max_leaf_nodes': leaves}))
    return result


def estimator(family, params):
    params = params.copy()
    params.pop('feature_mode', None)
    if family == 'Dummy':
        return DummyClassifier(strategy='prior')
    if family == 'Logistic':
        return LogisticRegression(**params, solver='liblinear', max_iter=700, random_state=SEED)
    if family == 'RandomForest':
        return RandomForestClassifier(**params, n_estimators=140, max_depth=16,
                                      max_features=.7, class_weight='balanced_subsample',
                                      n_jobs=3, random_state=SEED)
    if family == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(**params, max_iter=220, learning_rate=.07,
                                              min_samples_leaf=40, l2_regularization=3.,
                                              max_bins=128, early_stopping=False,
                                              random_state=SEED)
    raise ValueError(f'Unknown family: {family}')


def new_pipeline(family, params):
    return Pipeline([('preprocess', preprocessing(params.get('feature_mode', 'raw'))), ('model', estimator(family, params))])


def score_logit(scores):
    clipped = np.clip(np.asarray(scores), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrated_scores(pipeline, calibrator, x):
    raw = pipeline.predict_proba(x)[:, 1]
    return raw if calibrator is None else calibrator.predict_proba(score_logit(raw))[:, 1]


def grouped_errors(predictions, key):
    rows = []
    for group, d in predictions.groupby(key, observed=True, dropna=False):
        counts = d.outcome.value_counts()
        tp, fp, fn, tn = [int(counts.get(k, 0)) for k in ['TP', 'FP', 'FN', 'TN']]
        rows.append({'group': str(group), 'n': len(d), 'late': tp + fn, 'tp': tp,
                     'fp': fp, 'fn': fn, 'tn': tn,
                     'late_rate': (tp + fn) / len(d),
                     'precision': tp / (tp + fp) if tp + fp else np.nan,
                     'recall': tp / (tp + fn) if tp + fn else np.nan,
                     'false_positive_rate': fp / (fp + tn) if fp + tn else np.nan})
    return pd.DataFrame(rows)


def train_models(d, root):
    root = Path(root)
    out = root / 'outputs/tables'
    out.mkdir(parents=True, exist_ok=True)
    (root / 'outputs/models').mkdir(parents=True, exist_ok=True)
    config = json.loads((root / 'config.json').read_text())
    specs = model_specs()
    folds = list(temporal_folds(d, config))
    fold_audit, trials = [], []
    for number, start, end, train, validation in folds:
        fold_audit.append({'fold': number, 'validation_start': str(start), 'validation_end': str(end),
                           'train_n': len(train), 'validation_n': len(validation),
                           'train_late_rate': float(train.is_late.mean()),
                           'validation_late_rate': float(validation.is_late.mean())})
        require_available(validation, config['validation_start'])
        for name, family, params in specs:
            began = time.time()
            pipe = new_pipeline(family, params)
            pipe.fit(train[FEATURES], train.is_late.astype(int))
            scores = pipe.predict_proba(validation[FEATURES])[:, 1]
            row = {'fold': number, 'model': name, 'family': family,
                   'params': json.dumps(params), 'seconds': round(time.time() - began, 2),
                   **metrics(validation.is_late.astype(int), scores, .5)}
            trials.append(row)
            pd.DataFrame(trials).to_csv(out / 'temporal_cv_trials.csv', index=False)
            print(f'CV {number} {name}: AP={row["average_precision"]:.4f}', flush=True)
    pd.DataFrame(fold_audit).to_csv(out / 'temporal_cv_folds.csv', index=False)
    cv = pd.DataFrame(trials)
    summary = cv.groupby(['model', 'family'], as_index=False).agg(
        mean_ap=('average_precision', 'mean'), sd_ap=('average_precision', 'std'),
        worst_ap=('average_precision', 'min'), folds=('fold', 'nunique'))
    summary = summary.sort_values(['mean_ap', 'worst_ap', 'model'], ascending=[False, False, True])
    summary.to_csv(out / 'cv_summary.csv', index=False)
    winner = summary.iloc[0].model
    selected = []
    for family in summary.family.unique():
        selected.append(summary[summary.family.eq(family)].iloc[0].model)
    train, validation, test = final_split(d, config)
    midpoint = pd.Timestamp(config['calibration_end'])
    calibration = validation.loc[validation.order_approved_at.lt(midpoint)]
    threshold_data = validation.loc[validation.order_approved_at.ge(midpoint)]
    audit = []
    for name, frame in [('train', train), ('validation', validation), ('test', test),
                        ('calibration', calibration), ('threshold', threshold_data)]:
        if frame.empty or frame.is_late.nunique() != 2:
            raise ValueError(f'{name} requires both target classes')
        audit.append({'split': name, 'orders': len(frame), 'late': int(frame.is_late.sum()),
                      'late_rate': float(frame.is_late.mean()),
                      'approval_min': str(frame.order_approved_at.min()),
                      'approval_max': str(frame.order_approved_at.max()),
                      'delivery_max': str(frame.order_delivered_customer_date.max())})
        frame[['order_id', 'order_approved_at', 'is_late']].to_csv(out / f'{name}_membership.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'splits.csv', index=False)
    excluded = len(d) - len(train) - len(validation) - len(test)
    (out / 'purge.json').write_text(json.dumps({
        'approval_based_embargo_orders': excluded, 'purged_for_label_availability': 0,
        'embargo_days': config['embargo_days'], 'validation_start': config['validation_start'],
        'test_start': config['test_start'], 'policy': 'No individual arrival-time filtering'}))
    candidates = [s for s in specs if s[0] in selected]
    fitted, calibrators, thresholds, validation_rows = {}, {}, {}, []
    for name, family, params in candidates:
        pipe = Pipeline([('preprocess', preprocessing()), ('model', DummyClassifier(strategy='prior'))]) if family == 'Dummy' else new_pipeline(family, params)
        pipe.fit(train[FEATURES], train.is_late.astype(int))
        cal_raw = pipe.predict_proba(calibration[FEATURES])[:, 1]
        calibrator = None
        if family != 'Dummy':
            calibrator = LogisticRegression(C=1000., solver='lbfgs', max_iter=500, random_state=SEED)
            calibrator.fit(score_logit(cal_raw), calibration.is_late.astype(int))
            if calibrator.coef_[0, 0] <= 0:
                raise ValueError('Sigmoid calibration reversed ranking; investigate without test tuning')
        p_threshold = calibrated_scores(pipe, calibrator, threshold_data[FEATURES])
        threshold = .5 if family == 'Dummy' else choose_threshold(threshold_data.is_late.astype(int), p_threshold)
        fitted[name], calibrators[name], thresholds[name] = pipe, calibrator, threshold
        validation_rows.append({'model': name, 'params': json.dumps(params), 'threshold': threshold,
                                **metrics(threshold_data.is_late.astype(int), p_threshold, threshold)})
    pd.DataFrame(validation_rows).to_csv(out / 'validation_experiments.csv', index=False)
    selection = {'winner': winner, 'threshold': thresholds[winner], 'selection': config['selection_metric'],
                 'seed': SEED, 'features': FEATURES, 'refit': False, 'config': config,
                 'note': 'CV selects model; early validation calibrates; late validation sets threshold. Historical test period was inspected in v1.'}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(f'FROZEN {winner}; threshold={thresholds[winner]:.6f}; evaluating historical final period', flush=True)
    results, scores, all_scores, calibration_rows = [], {}, [], []
    yt = test.is_late.astype(int)
    for name, _, _ in candidates:
        raw = fitted[name].predict_proba(test[FEATURES])[:, 1]
        p = calibrated_scores(fitted[name], calibrators[name], test[FEATURES])
        scores[name] = p
        results.append({'model': name, 'threshold': thresholds[name], **metrics(yt, p, thresholds[name])})
        all_scores.append(pd.DataFrame({'order_id': test.order_id, 'model': name,
                                       'raw_score': raw, 'score': p, 'is_late': yt}))
        calibration_rows.extend([{'model': name, 'stage': 'raw', 'brier': brier_score_loss(yt, raw),
                                  'mean_score': float(raw.mean()), 'prevalence': float(yt.mean())},
                                 {'model': name, 'stage': 'calibrated', 'brier': brier_score_loss(yt, p),
                                  'mean_score': float(p.mean()), 'prevalence': float(yt.mean())}])
    pd.DataFrame(results).to_csv(out / 'test_metrics.csv', index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(out / 'all_test_scores.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(out / 'calibration_metrics.csv', index=False)
    model, calibrator = fitted[winner], calibrators[winner]
    p = scores[winner]
    predicted = p >= thresholds[winner]
    errors = test[['order_id', 'customer_state', 'main_category', 'planned_days', 'item_count',
                   'seller_count', 'physical_missing_share', 'is_late']].copy()
    errors['probability'] = p
    errors['predicted'] = predicted.astype(int)
    errors['outcome'] = np.select([(yt.to_numpy() == 1) & ~predicted,
                                  (yt.to_numpy() == 0) & predicted,
                                  (yt.to_numpy() == 1) & predicted], ['FN', 'FP', 'TP'], default='TN')
    errors.to_csv(out / 'test_predictions.csv', index=False)
    errors['planned_window'] = pd.cut(errors.planned_days, [-1, 7, 14, 21, 30, np.inf],
                                      labels=['0-7', '8-14', '15-21', '22-30', '31+'])
    errors['size_group'] = pd.cut(errors.item_count, [0, 1, 2, 4, np.inf], labels=['1', '2', '3-4', '5+'])
    for key in ['customer_state', 'planned_window', 'size_group', 'main_category']:
        grouped_errors(errors, key).to_csv(out / f'errors_by_{key}.csv', index=False)
    examples = pd.concat([errors.loc[errors.outcome.eq(k)].sort_values('probability', ascending=(k == 'FN')).head(5)
                          for k in ['FP', 'FN']])
    examples.to_csv(out / 'error_examples.csv', index=False)
    importance_x = test[FEATURES].sample(min(4000, len(test)), random_state=SEED)
    importance_y = yt.loc[importance_x.index]
    def ranking_score(est, x, y):
        return average_precision_score(y, calibrated_scores(est, calibrator, x))
    importance = permutation_importance(model, importance_x, importance_y, scoring=ranking_score,
                                        n_repeats=3, random_state=SEED, n_jobs=1)
    pd.DataFrame({'feature': FEATURES, 'ap_decrease': importance.importances_mean,
                  'std': importance.importances_std}).sort_values('ap_decrease', ascending=False).to_csv(out / 'permutation_importance.csv', index=False)
    rng, bootstrap = np.random.default_rng(SEED), []
    for _ in range(300):
        indices = rng.integers(0, len(yt), len(yt))
        m = metrics(yt.to_numpy()[indices], p[indices], thresholds[winner])
        bootstrap.append({k: m[k] for k in ['f1', 'recall', 'precision', 'average_precision']})
    pd.DataFrame(bootstrap).quantile([.025, .975]).to_csv(out / 'test_bootstrap_ci.csv')
    budget = []
    order = np.argsort(-p, kind='stable')
    for fraction in [.01, .05, .10, .20]:
        n = max(1, int(np.ceil(len(test) * fraction)))
        late = int(yt.to_numpy()[order[:n]].sum())
        budget.append({'review_fraction': fraction, 'review_orders': n, 'late_found': late,
                       'precision': late / n, 'recall': late / yt.sum()})
    pd.DataFrame(budget).to_csv(out / 'test_review_budget_diagnostic.csv', index=False)
    joblib.dump({'pipeline': model, 'calibrator': calibrator, 'threshold': thresholds[winner],
                 'features': FEATURES, 'config': config}, root / 'outputs/models/selected_model.joblib', compress=3)
    (out / 'environment.json').write_text(json.dumps({'python': platform.python_version(), 'pandas': pd.__version__,
                                                    'numpy': np.__version__, 'scikit_learn': sklearn.__version__}, indent=2))
    return {'splits': audit[:3], 'winner': winner, 'results': results, 'scores': scores,
            'test': test, 'purged': excluded}


def temporal_split(d):
    """Compatibility wrapper for tests and teaching examples."""
    config = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
    return final_split(d, config)
