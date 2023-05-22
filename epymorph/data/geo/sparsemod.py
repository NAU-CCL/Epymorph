from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.geo import Geo, validate_shape


def load() -> Geo:
    """
    With states from Pei model, implement the sparsemod movement model
    via Joe's dispersel kernel.
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
    humidity = load('humidity', np.double, (365, n))
    coords = load("coords", np.double, (n, 2))
    commuters = load("commuters", np.int_, (n, n))

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
    """
    Store probabilities of going from one subpopulation to the next.
    Since haversine function commutes (makes sense intuitibely), this
    will yield symmetric matrix.
    """
    dispersal_kernel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                coords_distance[i, j] = haversine(coords[i, :], coords[j, :])
                # Joe's function from sparsemodR paper
                dispersal_kernel[i, j] = 1 / \
                    (np.exp(coords_distance[i, j]*1/40))  # Insert \phi?
        # Normalize
        # TODO: move out of loop. use np.sum
        dispersal_kernel[i, ] = dispersal_kernel[i, ] / \
            sum(dispersal_kernel[i, ])
    print(coords_distance.shape)
    print(dispersal_kernel)
    """TODO: Check this dispersel kernel with Joe. I am changing a few things. Number one:
    I am using Haversine distance for my d_{i,j} instead of Euclidean distance. Made more 
    sense for lat long based distance function. Number two: I am not sure I am normalizing he 
    kernel the same way he does. It is hard for me to find how they did it in the cpp code."""

    validate_shape('dispersel_kernel', dispersal_kernel, (n, n))
    validate_shape('commuters', summed_commuters, (n,))

    return Geo(
        nodes=n,
        labels=labels,
        data={
            'population': population,
            'commuters': summed_commuters,
            'humidity': humidity,
            'dispersal_kernel': dispersal_kernel
        }
    )
