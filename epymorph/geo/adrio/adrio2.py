import functools
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from epymorph.data_shape import Shapes, SimDimensions
from epymorph.geography.scope import GeoScope
from epymorph.simulation import (AttributeDef, NamespacedAttributeResolver,
                                 SimulationFunction)

T_co = TypeVar('T_co', bound=np.generic, covariant=True)
"""The result type of an Adrio."""


class Adrio(SimulationFunction[NDArray[T_co]]):
    """
    ADRIO (or Abstract Data Resource Interface Object) are functions which are intended to
    load data from external sources for epymorph simulations. This may be from web APIs,
    local files or database, or anything imaginable.
    """

    def evaluate_in_context(
        self,
        data: NamespacedAttributeResolver,
        dim: SimDimensions,
        scope: GeoScope,
        rng: np.random.Generator,
    ) -> NDArray[T_co]:
        # TODO: use events for messaging?
        # print(f"Evaluating {self.__class__.__name__} ADRIO...")
        # t0 = perf_counter()
        value = super().evaluate_in_context(data, dim, scope, rng)
        # t1 = perf_counter()
        # print(f"Completed {self.__class__.__name__} ADRIO ({(t1 - t0):0.3f}s).")
        return value


AdrioClassT = TypeVar('AdrioClassT', bound=type[Adrio])


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
        else:
            # print(f"Using cached result for {self.__class__.__name__} ADRIO...")
            pass
        return cached_value

    cls.evaluate_in_context = evaluate_in_context
    return cls


class Scale(Adrio[np.float64]):
    """Scales the result of another ADRIO by multiplying its values by the configured factor."""

    _parent: Adrio[np.int64 | np.float64]
    _factor: float

    def __init__(self, parent: Adrio[np.int64 | np.float64], factor: float):
        self._parent = parent
        self._factor = factor

    def evaluate(self) -> NDArray[np.float64]:
        return self.defer(self._parent).astype(dtype=np.float64) * self._factor


class PopulationPerKm2(Adrio[np.float64]):
    """Calculates population density by combining the values from `population` and `land_area_km2`."""

    POPULATION = AttributeDef('population', int, Shapes.N)
    LAND_AREA_KM2 = AttributeDef('land_area_km2', float, Shapes.N)

    requirements = [POPULATION, LAND_AREA_KM2]

    def evaluate(self) -> NDArray[np.float64]:
        pop = self.data(self.POPULATION)
        area = self.data(self.LAND_AREA_KM2)
        return (pop / area).astype(dtype=np.float64)
