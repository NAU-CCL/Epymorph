from __future__ import annotations

from importlib.resources import as_file, files

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.geo import Geo, validate_shape


def load() -> Geo:
    """
    With states from Pei model, implement the sparsemod movement model
    via Joe's dispersel kernel. Output puts Pei populations, distances between them,
    and their humidity over time.
    """

    # Note: getting the dtype parameter and the return type of the
    # function to agree is more headache than it's worth; in large part
    # because numpy doesn't expose the type definitions we need to do so.
    # And the types are quickly lost when you do subsequent ops anyway.
    # See numpy.zeros for an example of how they do it internally.
    def read_file(name: str, dtype: DTypeLike, shape: tuple[int, ...]) -> NDArray:
        file = files('epymorph.data.geo').joinpath(f"pei-{name}.csv")
        with as_file(file) as f:
            data = np.loadtxt(f, delimiter=',', dtype=dtype)
            return validate_shape(name, data, shape)

    # Load base data:
    labels = ["FL", "GA", "MD", "NC", "SC", "VA"]
    n = len(labels)
    population = read_file('population', np.int_, (n,))
    humidity = read_file('humidity', np.double, (365, n))
    coords = read_file("coords", np.double, (n, 2))
    commuters = read_file("commuters", np.int_, (n, n))

    # Take row sum of commuters to get total movers by state
    summed_commuters = np.array([sum(commuters[i, ])
                                 for i in range(population.shape[0])])

    # Good measure for calculating distance between two points on a sphere
    def haversine(coord_one: NDArray, coord_two: NDArray) -> np.double:
        lat1 = coord_one[0]
        long1 = coord_one[1]
        lat2 = coord_two[0]
        long2 = coord_two[1]
        R = 3959.87433  # this is in miles.  For Earth radius in kilometers use 6372.8 km

        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(long2 - long1)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)

        a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
        c = 2*np.arcsin(np.sqrt(a))

        return R * c

    # Store haversine distances
    coords_distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                coords_distance[i, j] = haversine(coords[i, :], coords[j, :])

    validate_shape('commuters', summed_commuters, (n,))

    return Geo(
        nodes=n,
        labels=labels,
        data={
            'population': population,
            'commuters': summed_commuters,
            'humidity': humidity,
            'distances': coords_distance
        }
    )
