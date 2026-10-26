"""Approval-based cohorts and explicit label availability, without arrival-based filtering."""
import pandas as pd


def label_available_at(frame):
    """Punctual label is known on arrival; late label at start of first overdue day.

    This follows the date-based target: any arrival on the expected day is punctual.
    At the next midnight, an order still undelivered is necessarily late. The
    timestamp rule assumes the supplied milestone timestamps describe events
    correctly. The dataset has no full event log and eligibility is conditional
    on eventual delivery, as required by the assignment.
    """
    estimated = pd.to_datetime(frame['order_estimated_delivery_date'])
    actual = pd.to_datetime(frame['order_delivered_customer_date'])
    result = actual.where(frame['is_late'].eq(0), estimated.dt.normalize() + pd.Timedelta(days=1))
    return result


def require_available(frame, clock):
    """Fail the entire experiment if a cohort contains a not-yet-known label."""
    available = label_available_at(frame)
    bad = available.isna() | available.ge(pd.Timestamp(clock))
    if bad.any():
        raise ValueError(f'{int(bad.sum())} cohort labels unavailable at {clock}; redesign calendar boundaries')


def temporal_folds(frame, config):
    """Expanding training cohorts separated from one-month validation by fixed gap."""
    gap = pd.Timedelta(days=config['embargo_days'])
    for number, start in enumerate(config['fold_starts'], 1):
        start = pd.Timestamp(start)
        end = start + pd.DateOffset(months=config['fold_months'])
        train = frame.loc[frame.order_approved_at.lt(start - gap)].copy()
        validation = frame.loc[frame.order_approved_at.ge(start) & frame.order_approved_at.lt(end)].copy()
        require_available(train, start)
        yield number, start, end, train, validation


def final_split(frame, config):
    """All assignments depend solely on approval timestamps, never arrival speed."""
    start = pd.Timestamp(config['validation_start'])
    end = pd.Timestamp(config['validation_end'])
    test_start = pd.Timestamp(config['test_start'])
    train_end = start - pd.Timedelta(days=config['embargo_days'])
    train = frame.loc[frame.order_approved_at.lt(train_end)].copy()
    validation = frame.loc[frame.order_approved_at.ge(start) & frame.order_approved_at.lt(end)].copy()
    test = frame.loc[frame.order_approved_at.ge(test_start)].copy()
    require_available(train, start)
    require_available(validation, test_start)
    return train, validation, test
