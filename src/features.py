"""Deterministic transforms; these never learn statistics from validation or test."""
import numpy as np

SKEWED = ['total_price', 'total_freight', 'mean_weight_g', 'max_weight_g',
          'total_weight_g', 'mean_volume_cm3', 'payment_total',
          'approval_wait_hours', 'freight_share']


def log_cyclic_features(frame):
    """Compress nonnegative tails and represent hour/weekday/month circularly."""
    result = frame.copy()
    for column in SKEWED:
        result[column] = np.log1p(result[column].clip(lower=0))
    for column, period, origin in [('approval_hour', 24, 0),
                                    ('approval_weekday', 7, 0),
                                    ('approval_month', 12, 1)]:
        angle = 2 * np.pi * (result[column] - origin) / period
        result[column + '_sin'] = np.sin(angle)
        result[column + '_cos'] = np.cos(angle)
        result = result.drop(columns=column)
    return result
