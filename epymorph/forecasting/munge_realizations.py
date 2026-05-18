"""
This module extends the functionality of munge
over an additional axis, the realizations.
"""

from typing import Literal

import numpy as np

RealizationAggMethod = Literal["mean", "std", "quantiles"]
"""The supported methods for aggregating realizations."""


def quantile_method(q_val: float):
    def method(x):
        return x.quantile(q_val)

    method.__name__ = f"quantile_{round(100 * q_val, 1)}"
    return method


_every_025 = np.linspace(0.0, 1.0, 41)

"""Dictionaries of aggregation methods."""
agg_methods = {
    "mean": ["mean"],
    "std": ["std"],
    "median": ["median"],
    "quantiles": list(map(quantile_method, _every_025)),
}

"""Extend agg methods to individual quantiles.
   Supported quantiles are every 2.5%"""
for q_val in _every_025:
    agg_methods[f"quantile_{round(100 * q_val, 1)}"] = [quantile_method(q_val)]
