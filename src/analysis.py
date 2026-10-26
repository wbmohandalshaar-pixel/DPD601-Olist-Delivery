"""Descriptive populations, denominators, uncertainty and publication figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
sns.set_theme(style='whitegrid', palette=['#176B87', '#D55E00', '#6B8E23'], font_scale=1.05)

def rates(d, key):
    r = d.groupby(key, observed=True, dropna=False).is_late.agg(n='size', late='sum').reset_index()
    r['rate'] = r.late.astype(float) / r.n
    z = 1.96
    n = r.n
    p = r.rate
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    r['ci_low'] = center - half
    r['ci_high'] = center + half
    return r

def analyze(all_orders, d, root, result=None):
    figs = root / 'outputs/figures'
    tab = root / 'outputs/tables'
    figs.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(fig, name, kind, pop):
        fig.tight_layout()
        temporary = figs / f'{name}.tmp.png'
        fig.savefig(temporary, dpi=180, bbox_inches='tight')
        temporary.replace(figs / f'{name}.png')
        plt.close(fig)
        manifest.append(dict(figure=name, kind=kind, population=pop, n=len(d) if pop == 'Eligible delivered orders' else len(all_orders)))
    pop = 'Eligible delivered orders'
    counts = d.is_late.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    bars = ax.bar(['On time', 'Late'], counts, color=['#176B87', '#D55E00'])
    ax.bar_label(bars, labels=[f'{v:,} ({v / len(d):.1%})' for v in counts], padding=4)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set(ylabel='Orders', title='Delivered-order target distribution')
    save(fig, '01_target', 'bar', pop)
    a = all_orders.copy()
    a['month'] = a.order_purchase_timestamp.dt.to_period('M').astype(str)
    volume = a.groupby('month').size()
    e = d.copy()
    e['month'] = e.order_purchase_timestamp.dt.to_period('M').astype(str)
    monthly = rates(e, 'month')
    monthly.to_csv(tab / 'monthly_late.csv', index=False)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(volume.index, volume.values, marker='o', ms=3)
    axs[0].set(ylabel='All orders', title='Purchase month and delivery performance')
    eligible_months = monthly.loc[monthly.n.ge(200)]
    axs[1].plot(eligible_months.month, eligible_months.rate * 100, color='#D55E00', marker='o', ms=3)
    axs[1].fill_between(eligible_months.month, eligible_months.ci_low * 100, eligible_months.ci_high * 100, alpha=0.2, color='#D55E00')
    axs[1].set(ylabel='Late rate (%) for months n>=200', xlabel='Purchase month')
    axs[1].tick_params(axis='x', rotation=70)
    save(fig, '02_monthly', 'line', 'All orders and eligible delivered orders; separate denominators')
    fig, ax = plt.subplots(figsize=(8, 4.3))
    sns.histplot(data=d, x='planned_days', hue='is_late', bins=45, element='step', stat='density', common_norm=False, ax=ax)
    ax.set(title='Planned delivery window by observed outcome', xlabel='Planned calendar days from approval', ylabel='Density')
    save(fig, '03_planned_histogram', 'histogram', pop)
    state = rates(d, 'customer_state')
    state.to_csv(tab / 'state_late.csv', index=False)
    s = state[state.n >= 200].sort_values('rate')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(s.customer_state, s.rate * 100, xerr=np.vstack([(s.rate - s.ci_low) * 100, (s.ci_high - s.rate) * 100]), color='#176B87', capsize=2)
    ax.set_yticks(range(len(s)), [f'{r.customer_state} (n={r.n:,})' for r in s.itertuples()])
    ax.set(xlabel='Late rate (%) with 95% Wilson interval', title='Customer states with at least 200 eligible orders')
    save(fig, '04_state_rates', 'bar', pop)
    ss = rates(d, 'multi_seller')
    ss.to_csv(tab / 'multi_seller_late.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(['One seller', 'Multiple sellers'], ss.rate * 100)
    ax.set(ylabel='Late rate (%)', title='Seller multiplicity and observed delay')
    for i, r in ss.iterrows():
        ax.text(i, r.rate * 100 + 0.15, f'n={r.n:,}\n{r.rate:.1%}', ha='center')
    ax.set_ylim(0, ss.rate.max() * 135)
    save(fig, '05_multi_seller', 'bar', pop)
    cr = rates(d, 'main_category')
    cr.to_csv(tab / 'category_late.csv', index=False)
    c = cr.nlargest(12, 'n').sort_values('rate')
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(c.main_category.str.replace('_', ' '), c.rate * 100, xerr=np.vstack([(c.rate - c.ci_low) * 100, (c.ci_high - c.rate) * 100]), capsize=2)
    ax.set_yticks(range(len(c)), [f"{r.main_category.replace('_', ' ')} (n={r.n:,})" for r in c.itertuples()])
    ax.set(xlabel='Late rate (%) with 95% Wilson interval', title='Twelve largest order categories by volume')
    save(fig, '06_categories', 'bar', pop)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    sns.boxplot(data=d.assign(outcome=np.where(d.is_late.eq(1), 'Late', 'On time')), x='outcome', y='actual_delivery_days', showfliers=False, ax=ax)
    ax.set(title='Observed delivery duration by outcome', xlabel='Outcome', ylabel='Days from approval to customer delivery')
    save(fig, '07_delivery_box', 'boxplot', pop)
    sample = d.sample(min(4000, len(d)), random_state=601)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=sample, x='planned_days', y='actual_delivery_days', hue='is_late', alpha=0.4, s=15, ax=ax)
    ax.set(title=f'Planned and actual duration (random sample n={len(sample):,})', xlabel='Planned calendar days', ylabel='Actual elapsed days')
    save(fig, '08_planned_actual', 'scatter', pop)
    cols = ['item_count', 'seller_count', 'total_price', 'total_freight', 'total_weight_g', 'planned_days', 'approval_wait_hours', 'is_late']
    corr = d[cols].astype(float).corr(method='spearman')
    corr.to_csv(tab / 'spearman_correlations.csv')
    fig, ax = plt.subplots(figsize=(9, 6.2))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set(title='Spearman associations in eligible delivered orders')
    save(fig, '09_correlations', 'heatmap', pop)
    e = d.assign(physical_missing=np.where(d.physical_missing_share.gt(0), 'Some measurements missing', 'Measurements present'))
    miss = rates(e, 'physical_missing')
    miss.to_csv(tab / 'missingness_late.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    m = all_orders.groupby('eligibility_reason').size()
    axs[0].barh(m.index.str.replace('_', ' '), m.values)
    axs[0].set(xlabel='Orders', title='Exclusive eligibility waterfall')
    axs[1].bar(['Present', 'Some missing'], miss.set_index('physical_missing').reindex(['Measurements present', 'Some measurements missing']).rate * 100)
    axs[1].set(ylabel='Late rate (%)', title='Physical measurement missingness', ylim=(0, miss.rate.max() * 125))
    for i, r in enumerate(miss.set_index('physical_missing').reindex(['Measurements present', 'Some measurements missing']).itertuples()):
        axs[1].text(i, r.rate * 100 + 0.15, f'n={r.n:,}', ha='center')
    save(fig, '10_missing_exclusions', 'subplots', 'All orders and eligible delivered orders; separate denominators')
    e = d.assign(size_group=pd.cut(d.item_count, [0, 1, 2, 4, np.inf], labels=['1', '2', '3-4', '5+']))
    rates(e, 'size_group').to_csv(tab / 'size_late.csv', index=False)
    e = d.assign(route=np.select([d.same_state_share.eq(1), d.same_state_share.eq(0)], ['All same state', 'All interstate'], default='Mixed'))
    rates(e, 'route').to_csv(tab / 'route_late.csv', index=False)
    e = d.assign(planned_group=pd.cut(d.planned_days, [-1, 7, 14, 21, 30, np.inf], labels=['0-7', '8-14', '15-21', '22-30', '31+']))
    rates(e, 'planned_group').to_csv(tab / 'planned_window_late.csv', index=False)
    representation = all_orders.groupby('customer_state', dropna=False).agg(all_orders=('order_id', 'size'), eligible=('eligibility_reason', lambda x: x.eq('eligible').sum())).reset_index()
    representation['excluded_rate'] = 1 - representation.eligible / representation.all_orders
    representation.to_csv(tab / 'exclusion_representation.csv', index=False)
    difference = all_orders.payment_total - all_orders.total_price - all_orders.total_freight
    reconciliation = all_orders[['order_id', 'total_price', 'total_freight', 'payment_total']].copy()
    reconciliation['payment_minus_price_freight'] = difference
    reconciliation['both_observed'] = all_orders.has_items.eq(1) & all_orders.has_payments.eq(1)
    reconciliation.to_csv(tab / 'order_money_reconciliation.csv', index=False)
    e = d.assign(route=np.select([d.same_state_share.eq(1), d.same_state_share.eq(0)], ['All same state', 'All interstate'], default='Mixed'))
    pd.pivot_table(e, index='customer_state', columns='route', values='is_late', aggfunc=['mean', 'count'], observed=True).to_csv(tab / 'state_route_pivot.csv')
    statuses = all_orders.groupby('order_status').agg(orders=('order_id', 'size'), missing_approval=('order_approved_at', lambda x: x.isna().sum()))
    statuses.to_csv(tab / 'status_coverage.csv')
    if result:
        res = pd.DataFrame(result['results'])
        test = result['test']
        n = len(test)
        fig, axs = plt.subplots(1, len(res), figsize=(12, 3.8))
        for ax, r in zip(axs, res.to_dict('records')):
            cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
            sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues', ax=ax, xticklabels=['On time', 'Late'], yticklabels=['On time', 'Late'])
            ax.set(title=r['model'], xlabel='Predicted', ylabel='Actual')
        save(fig, '11_confusion_matrices', 'heatmap', 'Final chronological test orders')
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for name, prob in result['scores'].items():
            pr, re, _ = precision_recall_curve(test.is_late.astype(int), prob)
            ax.step(re, pr, where='post', label=name)
        ax.axhline(test.is_late.mean(), color='gray', linestyle='--', label=f'Prevalence {test.is_late.mean():.1%}')
        ax.legend()
        ax.set(xlabel='Recall', ylabel='Precision', title='Final test precision and recall curves')
        save(fig, '12_precision_recall', 'line', 'Final chronological test orders')
        imp = pd.read_csv(tab / 'permutation_importance.csv').head(10).sort_values('ap_decrease')
        fig, ax = plt.subplots(figsize=(9, 4.7))
        ax.barh(imp.feature, imp.ap_decrease, xerr=imp['std'])
        ax.set(title='Selected model permutation importance on final test', xlabel='Decrease in Average Precision (3 permutations; up to 4,000 rows)')
        save(fig, '13_importance', 'bar', 'Final chronological test orders')
    mf = pd.DataFrame(manifest)
    if result:
        mf.loc[mf.population.eq('Final chronological test orders'), 'n'] = len(result['test'])
    mf.to_csv(tab / 'figure_manifest.csv', index=False)
    summary = {'all_orders': len(all_orders), 'eligible_orders': len(d), 'late_orders': int(d.is_late.sum()), 'late_rate': float(d.is_late.mean()), 'all_purchase_min': str(all_orders.order_purchase_timestamp.min()), 'all_purchase_max': str(all_orders.order_purchase_timestamp.max())}
    import json
    (tab / 'summary.json').write_text(json.dumps(summary, indent=2))
