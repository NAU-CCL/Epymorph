import numpy as np
import pandas as pd
import pytest

from epymorph.forecasting.munge_realizations import (
    quantile_method,
    quantiles_from_intervals,
)


@pytest.mark.parametrize(
    ("quantile", "name", "expected"),
    [
        (0.0, "quantile_0.0", 0.0),
        (0.025, "quantile_2.5", 1.0),
        (0.5, "quantile_50.0", 20.0),
        (0.975, "quantile_97.5", 39.0),
        (1.0, "quantile_100.0", 40.0),
    ],
)
def test_quantile_method(quantile, name, expected):
    method = quantile_method(quantile)

    assert method.__name__ == name
    assert method(pd.Series([0, 10, 20, 30, 40])) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([], []),
        ([0], [("quantile_50.0", "quantile_50.0")]),
        ([100], [("quantile_0.0", "quantile_100.0")]),
        ([95], [("quantile_2.5", "quantile_97.5")]),
        (
            [50, 95, 50, 0],
            [
                ("quantile_2.5", "quantile_97.5"),
                ("quantile_25.0", "quantile_75.0"),
                ("quantile_25.0", "quantile_75.0"),
                ("quantile_50.0", "quantile_50.0"),
            ],
        ),
    ],
)
def test_quantiles_from_intervals(intervals, expected):
    original = intervals.copy()

    assert quantiles_from_intervals(intervals) == expected
    assert intervals == original


def test_quantiles_from_all_supported_intervals():
    intervals = list(range(0, 101, 5))

    assert quantiles_from_intervals(intervals) == [
        (f"quantile_{(100 - width) / 2:.1f}", f"quantile_{(100 + width) / 2:.1f}")
        for width in reversed(intervals)
    ]


@pytest.mark.parametrize(
    "interval", [-5, -0.1, 0.1, 2.5, 97.5, 100.1, 105, np.nan, np.inf, -np.inf]
)
def test_quantiles_from_intervals_rejects_invalid_widths(interval):
    with pytest.raises(ValueError, match="between 0 and 100 in increments of 5"):
        quantiles_from_intervals([50, interval])
