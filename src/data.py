"""Raw-data audit, relational checks, order-grain aggregation and eligible targets."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
FILES = {'orders': 'olist_orders_dataset.csv', 'customers': 'olist_customers_dataset.csv', 'items': 'olist_order_items_dataset.csv', 'payments': 'olist_order_payments_dataset.csv', 'products': 'olist_products_dataset.csv', 'sellers': 'olist_sellers_dataset.csv', 'translation': 'product_category_name_translation.csv'}
KEYS = {'orders': ['order_id'], 'customers': ['customer_id'], 'items': ['order_id', 'order_item_id'], 'payments': ['order_id', 'payment_sequential'], 'products': ['product_id'], 'sellers': ['seller_id'], 'translation': ['product_category_name']}
FORBIDDEN = {'order_status', 'order_delivered_customer_date', 'order_delivered_carrier_date', 'actual_delivery_days', 'delay_days', 'shipping_limit_date', 'order_id', 'customer_id', 'customer_unique_id', 'product_id', 'seller_id', 'is_late'}

def target(actual, estimated):
    """Nullable label; compare calendar dates rather than timestamp to midnight."""
    a = pd.to_datetime(actual, errors='coerce')
    e = pd.to_datetime(estimated, errors='coerce')
    y = (a.dt.normalize() > e.dt.normalize()).astype('Int64')
    return y.mask(a.isna() | e.isna())

def audit(tables, stage):
    rows = []
    for name, d in tables.items():
        duplicate_rows = int(d.duplicated().sum())
        for c in d:
            rows.append(dict(stage=stage, table=name, rows=len(d), columns=len(d.columns), column=c, dtype=str(d[c].dtype), missing=int(d[c].isna().sum()), missing_pct=100 * d[c].isna().mean(), duplicate_rows=duplicate_rows))
    return pd.DataFrame(rows)

def require_raw_files(raw):
    """Report all missing inputs before creating any generated artifacts."""
    raw = Path(raw)
    missing = [name for name in FILES.values() if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing original CSV files in ' + str(raw) + ':\n' +
            '\n'.join(missing) +
            '\nRestore the approved data package; Git intentionally excludes raw CSVs.'
        )


def load(raw):
    require_raw_files(raw)
    tables = {}
    hashes = {}
    for name, f in FILES.items():
        p = raw / f
        hashes[f] = hashlib.sha256(p.read_bytes()).hexdigest()
        tables[name] = pd.read_csv(p, dtype={c: 'string' for c in ['order_id', 'customer_id', 'customer_unique_id', 'product_id', 'seller_id', 'customer_zip_code_prefix', 'seller_zip_code_prefix']})
    return (tables, hashes)

def build(root):
    raw = root / 'data/raw'
    require_raw_files(raw)
    out = root / 'outputs/tables'
    out.mkdir(parents=True, exist_ok=True)
    (root / 'data/processed').mkdir(parents=True, exist_ok=True)
    t, hashes = load(raw)
    before = audit(t, 'before')
    issues = []
    joins = []
    for n, d in t.items():
        if d[KEYS[n]].isna().any().any() or d.duplicated(KEYS[n]).any():
            raise ValueError(f'Invalid primary key in {n}; investigate before proceeding')
    for child, fk, parent, pk in [('orders', 'customer_id', 'customers', 'customer_id'), ('items', 'order_id', 'orders', 'order_id'), ('payments', 'order_id', 'orders', 'order_id'), ('items', 'product_id', 'products', 'product_id'), ('items', 'seller_id', 'sellers', 'seller_id'), ('products', 'product_category_name', 'translation', 'product_category_name')]:
        missing = t[child][fk].notna() & ~t[child][fk].isin(t[parent][pk])
        issues.append(dict(check=f'{child}.{fk} -> {parent}.{pk}', count=int(missing.sum()), action='Keep untranslated categories as original Portuguese' if parent == 'translation' else 'Require zero orphan keys'))
        if parent != 'translation':
            assert not missing.any()
    for n in ['orders', 'items']:
        for c in t[n]:
            if 'date' in c or c in ['order_purchase_timestamp', 'order_approved_at']:
                old = t[n][c].notna()
                t[n][c] = pd.to_datetime(t[n][c], errors='coerce')
                issues.append(dict(check=f'{n}.{c} invalid timestamp', count=int((old & t[n][c].isna()).sum()), action='Coerce invalid dates to NaT; audit exclusions'))
    p = t['products']
    for c in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
        bad = p[c].notna() & p[c].le(0)
        issues.append(dict(check=f'{c} <= 0', count=int(bad.sum()), action='Set unknown; fit imputer on training only'))
        p.loc[bad, c] = np.nan
    for n, cols in [('items', ['price', 'freight_value']), ('payments', ['payment_value', 'payment_installments'])]:
        for c in cols:
            bad = t[n][c].lt(0)
            issues.append(dict(check=f'{n}.{c} negative', count=int(bad.sum()), action='Stop for investigation if any'))
            assert not bad.any()
    tr = t['translation']
    p = p.merge(tr, on='product_category_name', how='left', validate='many_to_one')
    p['category'] = p['product_category_name_english'].fillna(p['product_category_name']).fillna('unknown')
    p['volume_cm3'] = p['product_length_cm'] * p['product_height_cm'] * p['product_width_cm']
    it = t['items'].merge(p, on='product_id', how='left', validate='many_to_one').merge(t['sellers'], on='seller_id', how='left', validate='many_to_one')
    it = it.merge(t['orders'][['order_id', 'customer_id']], on='order_id', validate='many_to_one').merge(t['customers'][['customer_id', 'customer_state']], on='customer_id', validate='many_to_one')
    it['same_state'] = (it.seller_state == it.customer_state).astype(float)
    it['physical_missing'] = it[['product_weight_g', 'volume_cm3']].isna().any(axis=1).astype(int)
    g = it.groupby('order_id', sort=False)
    agg = g.agg(item_count=('order_item_id', 'size'), seller_count=('seller_id', 'nunique'), product_count=('product_id', 'nunique'), category_count=('category', 'nunique'), total_price=('price', 'sum'), total_freight=('freight_value', 'sum'), mean_weight_g=('product_weight_g', 'mean'), max_weight_g=('product_weight_g', 'max'), mean_volume_cm3=('volume_cm3', 'mean'), same_state_share=('same_state', 'mean'), physical_missing_share=('physical_missing', 'mean'))
    agg['total_weight_g'] = g.product_weight_g.sum(min_count=1)
    cats = it.groupby(['order_id', 'category']).size().rename('n').reset_index().sort_values(['order_id', 'n', 'category'], ascending=[True, False, True]).drop_duplicates('order_id').set_index('order_id').category
    agg['main_category'] = cats
    pay = t['payments']
    pg = pay.groupby('order_id', sort=False)
    pa = pg.agg(payment_count=('payment_sequential', 'size'), payment_total=('payment_value', 'sum'), max_installments=('payment_installments', 'max'), payment_type_count=('payment_type', 'nunique'))
    pt = pay.groupby(['order_id', 'payment_type']).payment_value.sum().reset_index().sort_values(['order_id', 'payment_value', 'payment_type'], ascending=[True, False, True]).drop_duplicates('order_id').set_index('order_id').payment_type
    pa['main_payment_type'] = pt
    d = t['orders'].copy()
    for name, right, key in [('customers', t['customers'], 'customer_id'), ('item_aggregates', agg.reset_index(), 'order_id'), ('payment_aggregates', pa.reset_index(), 'order_id')]:
        n = len(d)
        d = d.merge(right, on=key, how='left', validate='many_to_one' if key == 'customer_id' else 'one_to_one')
        joins.append(dict(join=name, before=n, after=len(d), unmatched=int((~d[key].isin(right[key])).sum())))
        assert len(d) == n
    assert np.isclose(d.total_price.sum(), t['items'].price.sum())
    assert np.isclose(d.total_freight.sum(), t['items'].freight_value.sum())
    assert np.isclose(d.payment_total.sum(), pay.payment_value.sum())
    d['has_items'] = d.item_count.notna().astype(int)
    d['has_payments'] = d.payment_count.notna().astype(int)
    d['approval_hour'] = d.order_approved_at.dt.hour
    d['approval_weekday'] = d.order_approved_at.dt.dayofweek
    d['approval_month'] = d.order_approved_at.dt.month
    d['approval_wait_hours'] = (d.order_approved_at - d.order_purchase_timestamp).dt.total_seconds() / 3600
    d['planned_days'] = (d.order_estimated_delivery_date.dt.normalize() - d.order_approved_at.dt.normalize()).dt.days
    d['freight_share'] = np.divide(d.total_freight, d.total_price.where(d.total_price.gt(0)))
    d['multi_seller'] = d.seller_count.gt(1).astype(int)
    d['actual_delivery_days'] = (d.order_delivered_customer_date - d.order_approved_at).dt.total_seconds() / 86400
    d['delay_days'] = (d.order_delivered_customer_date.dt.normalize() - d.order_estimated_delivery_date.dt.normalize()).dt.days
    d['is_late'] = target(d.order_delivered_customer_date, d.order_estimated_delivery_date)
    reasons = pd.Series('eligible', index=d.index)
    rules = [('not_delivered', d.order_status.ne('delivered')), ('missing_delivery_or_estimate', d.order_delivered_customer_date.isna() | d.order_estimated_delivery_date.isna()), ('missing_purchase_or_approval', d.order_purchase_timestamp.isna() | d.order_approved_at.isna()), ('approval_before_purchase', d.order_approved_at.lt(d.order_purchase_timestamp)), ('delivered_before_approval', d.order_delivered_customer_date.lt(d.order_approved_at)), ('estimate_before_approval_day', d.planned_days.lt(0))]
    for reason, bad in rules:
        reasons.loc[reasons.eq('eligible') & bad] = reason
        issues.append(dict(check=reason, count=int(bad.sum()), action='First applicable reason in exclusive exclusion waterfall'))
    d['eligibility_reason'] = reasons
    eligible = d.loc[reasons.eq('eligible')].copy()
    assert eligible.is_late.notna().all()
    for a, b in [('order_delivered_carrier_date', 'order_approved_at'), ('order_delivered_customer_date', 'order_delivered_carrier_date')]:
        issues.append(dict(check=f'{a} earlier than {b}', count=int(d[a].lt(d[b]).sum()), action='Diagnostic only; not used as predictive features or exclusion rule'))
    pd.concat([before, audit(t, 'after')]).to_csv(out / 'column_audit.csv', index=False)
    pd.DataFrame(issues).to_csv(out / 'quality_checks.csv', index=False)
    pd.DataFrame(joins).to_csv(out / 'join_audit.csv', index=False)
    reasons.value_counts().rename_axis('reason').rename('orders').to_csv(out / 'exclusions.csv')
    (out / 'raw_sha256.json').write_text(json.dumps(hashes, indent=2))
    for name, frame in [('all_orders', d), ('model_orders', eligible)]:
        path = root / f'data/processed/{name}.csv'
        temporary = path.with_suffix('.tmp')
        frame.to_csv(temporary, index=False)
        if len(pd.read_csv(temporary, usecols=['order_id'])) != len(frame):
            raise IOError(f'Incomplete CSV output: {name}')
        temporary.replace(path)
    return (d, eligible, t)
NUMERIC = ['item_count', 'seller_count', 'product_count', 'category_count', 'total_price', 'total_freight', 'mean_weight_g', 'max_weight_g', 'total_weight_g', 'mean_volume_cm3', 'same_state_share', 'physical_missing_share', 'payment_count', 'payment_total', 'max_installments', 'payment_type_count', 'has_items', 'has_payments', 'approval_hour', 'approval_weekday', 'approval_month', 'approval_wait_hours', 'planned_days', 'freight_share', 'multi_seller']
CATEGORICAL = ['customer_state', 'main_category', 'main_payment_type']
FEATURES = NUMERIC + CATEGORICAL
assert not set(FEATURES) & FORBIDDEN
