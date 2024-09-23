"""
Implements the base class for all ADRIOs, as well as some general-purpose
ADRIO implementations.
"""

import functools
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_shape import Shapes
from epymorph.data_usage import DataEstimate
from epymorph.simulation import AttributeDef, SimulationFunction

T_co = TypeVar("T_co", bound=np.generic, covariant=True)
"""The result type of an Adrio."""


class Adrio(SimulationFunction[NDArray[T_co]]):
    """
    ADRIO (or Abstract Data Resource Interface Object) are functions which are intended
    to load data from external sources for epymorph simulations. This may be from
    web APIs, local files or database, or anything imaginable.
    """

    @property
    def full_name(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def estimate_data(self) -> DataEstimate | None:
        """Estimate the data usage of this ADRIO in a RUME.
        If a reasonable estimate cannot be made, None is returned."""
        return None


AdrioClassT = TypeVar("AdrioClassT", bound=type[Adrio])


def adrio_cache(cls: AdrioClassT) -> AdrioClassT:
    """Adrio class decorator to add result-caching behavior."""

    original_eval = cls.evaluate_in_context
    cached_value: NDArray | None = None
    cached_hash: int | None = None

    @functools.wraps(original_eval)
    def evaluate_in_context(self, data, dim, scope, rng):
        req_hashes = (data.resolve(r).data.tobytes() for r in self.requirements)
        curr_hash = hash(tuple([dim, scope, *req_hashes]))
        nonlocal cached_value, cached_hash
        if cached_value is None or cached_hash != curr_hash:
            cached_value = original_eval(self, data, dim, scope, rng)
            cached_hash = curr_hash
        return cached_value

    cls.evaluate_in_context = evaluate_in_context
    return cls


class NodeId(Adrio[np.str_]):
    """An ADRIO that provides the node IDs as they exist in the geo scope."""

    @override
    def evaluate(self) -> NDArray:
        return self.scope.get_node_ids()


class Scale(Adrio[np.float64]):
    """Scales the result of another ADRIO by multiplying values by the given factor."""

    _parent: Adrio[np.int64 | np.float64]
    _factor: float

    def __init__(self, parent: Adrio[np.int64 | np.float64], factor: float):
        self._parent = parent
        self._factor = factor

    @override
    def evaluate(self) -> NDArray[np.float64]:
        return self.defer(self._parent).astype(dtype=np.float64) * self._factor


class PopulationPerKm2(Adrio[np.float64]):
    """
    Calculates population density by combining the values from attributes named
    `population` and `land_area_km2`. You must provide those attributes
    separately.
    """

    POPULATION = AttributeDef("population", int, Shapes.N)
    LAND_AREA_KM2 = AttributeDef("land_area_km2", float, Shapes.N)

    requirements = [POPULATION, LAND_AREA_KM2]

    @override
    def evaluate(self) -> NDArray[np.float64]:
        pop = self.data(self.POPULATION)
        area = self.data(self.LAND_AREA_KM2)
        return (pop / area).astype(dtype=np.float64)
