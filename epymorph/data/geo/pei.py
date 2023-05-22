from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.geo import Geo, validate_shape


def load() -> Geo:
    """
    Pei-style geo model. Defines nodes representing 6 east-coast US states, each with:
    - total population
    - state-to-state commuters
    - absolute humidity by day (for 365 days)
    """

    # Note: getting the dtype parameter and the return type of the
    # function to agree is more headache than it's worth; in large part
    # because numpy doesn't expose the type definitions we need to do so.
    # And the types are quickly lost when you do subsequent ops anyway.
    # See numpy.zeros for an example of how they do it internally.
    def load(name: str, dtype: DTypeLike, shape: tuple[int, ...]) -> NDArray:
        data = np.loadtxt(
            f"./epymorph/data/geo/pei-{name}.csv", delimiter=',', dtype=dtype)
        return validate_shape(name, data, shape)

    # Load base data:
    labels = ["FL", "GA", "MD", "NC", "SC", "VA"]
    n = len(labels)
    population = load('population', np.int_, (n,))
    commuters = load('commuters', np.int_, (n, n))
    humidity = load('humidity', np.double, (365, n))
    coords = load("coords", np.double, (n, 2))

    # Precompute data views:
    # Average commuters between node pairs.
    commuters_average = np.zeros(commuters.shape, dtype=np.double)
    for i in range(commuters.shape[0]):
        for j in range(i + 1, commuters.shape[1]):
            nbar = (commuters[i, j] + commuters[j, i]) // 2
            commuters_average[i, j] = nbar
            commuters_average[j, i] = nbar
    # Total commuters living in each state.
    commuters_by_node = commuters.sum(axis=1, dtype=np.int_)
    # Commuters as a ratio to the total commuters living in that state.
    commuting_probability = commuters / commuters_by_node[:, None]

    validate_shape('commuters_average', commuters_average, (n, n))
    validate_shape('commuters_by_node', commuters_by_node, (n,))
    validate_shape('commuting_probability', commuting_probability, (n, n))

    return Geo(
        nodes=n,
        labels=labels,
        data={
            'population': population,
            'commuters': commuters,
            'commuters_average': commuters_average,
            'commuters_by_node': commuters_by_node,
            'commuting_probability': commuting_probability,
            'humidity': humidity
        }
    )
