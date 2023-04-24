from __future__ import annotations

import numpy as np

from epymorph.geo import Geo


def load_geo() -> Geo:
    """
    Pei-style geo model. Defines nodes representing 6 east-coast US states, each with:
    - total population
    - state-to-state commuters
    - absolute humidity by day (for 365 days)
    """
    def load_data(name, dtype):
        return np.loadtxt(f"./data/pei-{name}.csv", delimiter=',', dtype=dtype)
    return Geo(
        nodes=6,
        labels=["FL", "GA", "MD", "NC", "SC", "VA"],
        data={
            'population': load_data('population', np.int_),
            'commuters': load_data('commuters', np.int_),
            'humidity': load_data('humidity', np.double)
        }
    )
