"""
Implements the base class for all ADRIOs, as well as some general-purpose
ADRIO implementations.
"""

import functools
from abc import abstractmethod
from time import perf_counter
from typing import Callable, TypeVar, final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from epymorph.data_shape import Shapes, SimDimensions
from epymorph.data_type import AttributeArray
from epymorph.data_usage import DataEstimate, EmptyDataEstimate
from epymorph.database import (
    NAMESPACE_PLACEHOLDER,
    AbsoluteName,
    DataResolver,
    ModuleNamespace,
    evaluate_param,
)
from epymorph.event import AdrioProgress, DownloadActivity, EventBus
from epymorph.geography.scope import GeoScope
from epymorph.simulation import AttributeDef, SimulationFunction

T_co = TypeVar("T_co", bound=np.generic, covariant=True)
"""The result type of an Adrio."""

ProgressCallback = Callable[[float, DownloadActivity | None], None]

_events = EventBus()


class Adrio(SimulationFunction[NDArray[T_co]]):
    """
    ADRIO (or Abstract Data Resource Interface Object) are functions which are intended
    to load data from external sources for epymorph simulations. This may be from
    web APIs, local files or database, or anything imaginable.
    """

    @property
    def full_name(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def estimate_data(self) -> DataEstimate:
        """Estimate the data usage of this ADRIO in a RUME.
        If a reasonable estimate cannot be made, None is returned."""
        return EmptyDataEstimate(self.full_name)

    @abstractmethod
    def evaluate_adrio(self) -> NDArray[T_co]:
        """
        Implement this method to provide logic for the function.
        Your implementation is free to use `data`, `dim`, and `rng`.
        You can also use `defer` to utilize another SimulationFunction instance.
        """

    @override
    @final
    def evaluate(self) -> NDArray[T_co]:
        """The ADRIO parent class overrides this to provide ADRIO-specific
        functionality. ADRIO implementations should override `evaluate_adrio`."""
        _events.on_adrio_progress.publish(
            AdrioProgress(
                adrio_name=self.full_name,
                final=False,
                ratio_complete=0,
                download=None,
                duration=None,
            )
        )
        t0 = perf_counter()
        result = self.evaluate_adrio()
        t1 = perf_counter()
        _events.on_adrio_progress.publish(
            AdrioProgress(
                adrio_name=self.full_name,
                final=True,
                ratio_complete=1,
                download=None,
                duration=t1 - t0,
            )
        )
        return result

    @final
    def progress(
        self,
        ratio_complete: float,
        download: DownloadActivity | None = None,
    ) -> None:
        """Emit a progress event."""
        _events.on_adrio_progress.publish(
            AdrioProgress(
                adrio_name=self.full_name,
                final=False,
                ratio_complete=ratio_complete,
                download=download,
                duration=None,
            )
        )


@evaluate_param.register
def _(
    value: Adrio,
    name: AbsoluteName,
    data: DataResolver,
    dim: SimDimensions | None,
    scope: GeoScope | None,
    rng: np.random.Generator | None,
) -> AttributeArray:
    # depth-first evaluation guarantees `data` has our dependencies.
    namespace = name.to_namespace()
    sim_func = value.with_context_internal(namespace, data, dim, scope, rng)
    return np.asarray(sim_func.evaluate())


AdrioClassT = TypeVar("AdrioClassT", bound=type[Adrio])


def adrio_cache(cls: AdrioClassT) -> AdrioClassT:
    """Adrio class decorator to add result-caching behavior."""

    orig_with_context = cls.with_context_internal
    cached_instance: AdrioClassT | None = None
    cached_hash: int | None = None

    @functools.wraps(orig_with_context)
    def with_context_internal(
        self,
        namespace: ModuleNamespace = NAMESPACE_PLACEHOLDER,
        data: DataResolver | None = None,
        dim: SimDimensions | None = None,
        scope: GeoScope | None = None,
        rng: np.random.Generator | None = None,
    ):
        if data is None:
            req_hashes = ()
        else:
            req_hashes = (
                data.resolve(namespace.to_absolute(r.name), r).tobytes()
                for r in self.requirements
            )
        curr_hash = hash(tuple([str(namespace), dim, scope, *req_hashes]))
        nonlocal cached_instance, cached_hash
        if cached_instance is None or cached_hash != curr_hash:
            cached_instance = orig_with_context(self, namespace, data, dim, scope, rng)
            cached_hash = curr_hash
        return cached_instance

    cls.with_context_internal = with_context_internal
    return cls


class NodeId(Adrio[np.str_]):
    """An ADRIO that provides the node IDs as they exist in the geo scope."""

    @override
    def evaluate_adrio(self) -> NDArray:
        return self.scope.node_ids


class Scale(Adrio[np.float64]):
    """Scales the result of another ADRIO by multiplying values by the given factor."""

    _parent: Adrio[np.int64 | np.float64]
    _factor: float

    def __init__(self, parent: Adrio[np.int64 | np.float64], factor: float):
        """
        Initializes scaling with the ADRIO to be scaled and with the factor to multiply
        those resulting ADRIO values by.

        Parameters
        ----------
        parent : Adrio[np.int64 | np.float64]
            The ADRIO to scale all values for.
        factor : float
            The factor to multiply all resulting ADRIO values by.
        """
        self._parent = parent
        self._factor = factor

    @override
    def evaluate_adrio(self) -> NDArray[np.float64]:
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
    def evaluate_adrio(self) -> NDArray[np.float64]:
        pop = self.data(self.POPULATION)
        area = self.data(self.LAND_AREA_KM2)
        return (pop / area).astype(dtype=np.float64)
