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

    # Load base data:
    def load_data(name, dtype):
        return np.loadtxt(f"./data/pei-{name}.csv", delimiter=',', dtype=dtype)

    population = load_data('population', np.int_)
    commuters = load_data('commuters', np.int_)
    humidity = load_data('humidity', np.double)

    # Precompute data views:
    # Average commuters between node pairs.
    commuters_average = np.zeros(commuters.shape)
    for i in range(commuters.shape[0]):
        for j in range(i + 1, commuters.shape[1]):
            nbar = (commuters[i, j] + commuters[j, i]) // 2
            commuters_average[i, j] = nbar
            commuters_average[j, i] = nbar
    # Total commuters living in each state.
    commuters_by_state = commuters.sum(axis=1, dtype=np.int_)
    # Commuters as a ratio to the total commuters living in that state.
    commuting_probability = commuters / commuters_by_state[:, None]

    return Geo(
        nodes=6,
        labels=["FL", "GA", "MD", "NC", "SC", "VA"],
        data={
            'population': population,
            'commuters': commuters,
            'commuters_average': commuters_average,
            'commuters_by_state': commuters_by_state,
            'commuting_probability': commuting_probability,
            'humidity': humidity
        }
    )
