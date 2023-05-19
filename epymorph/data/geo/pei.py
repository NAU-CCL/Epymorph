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

    def haversine(coord_one: NDArray, coord_two: NDArray) -> NDArray:
        x1 = coord_one[0]
        x2 = coord_one[1]
        y1 = coord_two[0]
        y2 = coord_two[1]
        return 2*np.arcsin(np.sqrt((np.sin((x1 - y1)/2))**2 + np.cos(x1)*np.cos(y1)*(np.sin((x2 - y2)/2))**2))

    coords_distance = np.zeros((n,n))
    dispersal_kernel = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                coords_distance[i, j] = haversine(coords[i, :], coords[j, :])
                dispersal_kernel[i, j] = 1/(np.exp(coords_distance[i,j])) # Insert \phi? 
        dispersal_kernel[i, ] = dispersal_kernel[i, ]/sum(dispersal_kernel[i, ])
        """TODO: Check this dispersel kernel with Joe. I am changing a few things. Number one:
        I am using Haversine distance for my d_{i,j} instead of Euclidean distance. Made more 
        sense for lat long based distance function. Number two: I am not sure I am normalizing he 
        kernel the same way he does. It is hard for me to find how they did it in the cpp code."""

    print(sum(dispersal_kernel[1, ]))
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
            'humidity': humidity,
            'dispersal_kernel': dispersal_kernel
        }
    )
