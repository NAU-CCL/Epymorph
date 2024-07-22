from time import perf_counter
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
    """yo, adrio-n"""

    # TODO: actually I'm not sure caching is a good idea...
    # what if a dependency changes but the scope doesn't?
    # Or if there's randomness? Can't guarantee all future ADRIOs
    # aren't using these features, unless we limit what ADRIOs can do.
    # (Which would make them a slightly different animal from SimFunctions.)
    _cached_value: NDArray[T_co] | None = None
    _cached_scope: GeoScope | None = None

    def evaluate_in_context(
        self,
        data: NamespacedAttributeResolver,
        dim: SimDimensions,
        scope: GeoScope,
        rng: np.random.Generator,
    ) -> NDArray[T_co]:
        value = self._cached_value
        cached_scope = self._cached_scope
        if value is None or scope != cached_scope:
            # TODO: use events for messaging?
            print(f"Evaluating {self.__class__.__name__} ADRIO...")
            t0 = perf_counter()
            value = super().evaluate_in_context(data, dim, scope, rng)
            self._cached_value = value
            self._cached_scope = scope
            t1 = perf_counter()
            print(f"Completed {self.__class__.__name__} ADRIO ({(t1 - t0):0.3f}s).")
        return value


class Scale(Adrio[np.float64]):

    _parent: Adrio[np.int64 | np.float64]
    _factor: float

    def __init__(self, parent: Adrio[np.int64 | np.float64], factor: float):
        self._parent = parent
        self._factor = factor

    def evaluate(self) -> NDArray[np.float64]:
        return self.defer(self._parent).astype(dtype=np.float64) * self._factor


class PopulationPerKm2(Adrio[np.float64]):
    POPULATION = AttributeDef('population', int, Shapes.N)
    LAND_AREA_KM2 = AttributeDef('land_area_km2', float, Shapes.N)

    requirements = [POPULATION, LAND_AREA_KM2]

    def evaluate(self) -> NDArray[np.float64]:
        pop = self.data(self.POPULATION)
        area = self.data(self.LAND_AREA_KM2)
        return (pop / area).astype(dtype=np.float64)
