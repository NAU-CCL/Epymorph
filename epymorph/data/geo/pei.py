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

    return Geo(
        nodes=n,
        labels=labels,
        data={
            'population': population,
            'commuters': commuters,
            'humidity': humidity
        }
    )
