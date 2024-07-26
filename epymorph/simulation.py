"""General simulation requisites and utility functions."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from importlib import reload
from typing import (Any, Callable, Generator, Generic, Iterable, NamedTuple,
                    Self, Sequence, TypeVar, final, overload)

import numpy as np
from numpy.random import SeedSequence
from numpy.typing import NDArray

from epymorph.data_shape import DataShape, Shapes, SimDimensions
from epymorph.data_type import (AttributeArray, AttributeType, AttributeValue,
                                dtype_as_np, dtype_check)
from epymorph.database import (AbsoluteName, AttributeName, Database,
                               ModuleNamespace)
from epymorph.error import AttributeException
from epymorph.util import acceptable_name


def default_rng(seed: int | SeedSequence | None = None) -> Callable[[], np.random.Generator]:
    """
    Convenience constructor to create a factory function for a simulation's random number generator,
    optionally with a given seed.
    """
    return lambda: np.random.default_rng(seed)


def enable_logging(filename: str = 'debug.log', movement: bool = True) -> None:
    """Enable simulation logging to file."""
    reload(logging)
    logging.basicConfig(filename=filename, filemode='w')
    if movement:
        logging.getLogger('movement').setLevel(logging.DEBUG)


########
# Time #
########


@dataclass(frozen=True)
class TimeFrame:
    """The time frame of a simulation."""

    @classmethod
    def of(cls, start_date_iso8601: str, duration_days: int) -> Self:
        """Alternate constructor for TimeFrame, parsing start date from an ISO-8601 string."""
        return cls(date.fromisoformat(start_date_iso8601), duration_days)

    start_date: date
    """The first date in the simulation."""
    duration_days: int
    """The number of days for which to run the simulation."""

    @cached_property
    def end_date(self) -> date:
        """The last date included in the simulation."""
        return self.start_date + timedelta(days=self.duration_days)

    def is_subset(self, other: 'TimeFrame') -> bool:
        """Is the given TimeFrame a subset of this one?"""
        return self.start_date <= other.start_date and self.end_date >= other.end_date


class Tick(NamedTuple):
    """
    A Tick bundles related time-step information. For instance, each time step corresponds to a calendar day,
    a numeric day (i.e., relative to the start of the simulation), which tau step this corresponds to, and so on.
    """
    sim_index: int
    """Which simulation step are we on? (0,1,2,3,...)"""
    day: int
    """Which day increment are we on? Same for each tau step: (0,0,1,1,2,2,...)"""
    date: date
    """The calendar date corresponding to `day`"""
    step: int
    """Which tau step are we on? (0,1,0,1,0,1,...)"""
    tau: float
    """What's the tau length of the current step? (0.666,0.333,0.666,0.333,...)"""


class TickDelta(NamedTuple):
    """
    An offset relative to a Tick expressed as a number of days which should elapse,
    and the step on which to end up. In applying this delta, it does not matter which
    step we start on. We need the Clock configuration to apply a TickDelta, so see
    Clock for the relevant method.
    """
    days: int  # number of whole days
    step: int  # which tau step within that day (zero-indexed)


NEVER = TickDelta(-1, -1)
"""
A special TickDelta value which expresses an event that should never happen.
Any Tick plus Never returns Never.
"""


def resolve_tick_delta(dim: SimDimensions, tick: Tick, delta: TickDelta) -> int:
    """Add a delta to a tick to get the index of the resulting tick."""
    return -1 if delta.days == -1 else \
        tick.sim_index - tick.step + (dim.tau_steps * delta.days) + delta.step


def simulation_clock(dim: SimDimensions) -> Iterable[Tick]:
    """Generator for the sequence of ticks which makes up the simulation clock."""
    one_day = timedelta(days=1)
    tau_steps = list(enumerate(dim.tau_step_lengths))
    curr_index = 0
    curr_date = dim.start_date
    for day in range(dim.days):
        for step, tau in tau_steps:
            yield Tick(curr_index, day, curr_date, step, tau)
            curr_index += 1
        curr_date += one_day


############################
# Attributes and resolvers #
############################


AttributeT = TypeVar('AttributeT', bound=AttributeType)
"""The data type of an attribute; maps to the numpy type of the attribute array."""

# NOTE: I had AttributeT as covariant originally but that seems to cause pyright some headache
# interpreting typed method overloads (which is practically the whole point of making AttributeKey generic).
# So for now this must be invariant, but I think that's okay. Rather than being able to say things like:
# `AttributeKey[type[int] | type[float]]` you have to say `AttributeKey[type[int]] | AttributeKey[type[float]]`.


@dataclass(frozen=True)
class AttributeKey(Generic[AttributeT]):
    """The identity of a simulation attribute."""
    name: str
    type: AttributeT
    shape: DataShape

    def __post_init__(self):
        if acceptable_name.match(self.name) is None:
            raise ValueError(f"Invalid attribute name: {self.name}")
        try:
            dtype_as_np(self.type)
        except Exception as e:
            msg = f"AttributeDef's type is not correctly specified: {self.type}\n" \
                + "See documentation for appropriate type designations."
            raise ValueError(msg) from e
        object.__setattr__(self, 'attribute_name', AttributeName(self.name))

    @overload
    def dtype(self: "AttributeKey[type[int]]") -> np.dtype[np.int64]: ...
    @overload
    def dtype(self: "AttributeKey[type[float]]") -> np.dtype[np.float64]: ...
    @overload
    def dtype(self: "AttributeKey[type[str]]") -> np.dtype[np.str_]: ...
    # providing overloads for structured types is basically impossible without mapped types,
    # so callers are on their own for that.

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of this attribute in a numpy-equivalent type."""
        return dtype_as_np(self.type)


@dataclass(frozen=True)
class AttributeDef(AttributeKey[AttributeT]):
    """Definition of a simulation attribute; the identity plus optional default value and comment."""
    name: str
    type: AttributeT
    shape: DataShape
    default_value: AttributeValue | None = field(default=None, compare=False)
    comment: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if acceptable_name.match(self.name) is None:
            raise ValueError(f"Invalid attribute name: {self.name}")
        try:
            dtype_as_np(self.type)
        except Exception as e:
            msg = f"AttributeDef's type is not correctly specified: {self.type}\n" \
                + "See documentation for appropriate type designations."
            raise ValueError(msg) from e
        if self.default_value is not None and not dtype_check(self.type, self.default_value):
            msg = "AttributeDef's default value does not align with its dtype."
            raise ValueError(msg)


class _BaseAttributeResolver:
    """Base class for attribute resolvers."""

    _data: Database[AttributeArray]
    _dim: SimDimensions

    def __init__(self, data: Database[AttributeArray], dim: SimDimensions):
        self._data = data
        self._dim = dim

    def _resolve(self, attr: tuple[AbsoluteName, AttributeKey]) -> NDArray:
        """Resolve an attribute value by AbsoluteName and convert it to the type and shape in the given AttributeKey."""
        name, attr_key = attr
        matched = self._data.query(name)
        if matched is None:
            msg = f"Missing attribute '{name}'"
            raise AttributeException(msg)

        # Assume that we've already validated the attributes, so we don't have to do that every time.
        # In standard simulation workflows, we should therefore not see misses or incompatibilities.
        # But they are possible in use-cases outside these workflows.

        try:
            value = matched.value
            value = value.astype(attr_key.dtype, casting='safe',
                                 subok=False, copy=False)
            value = attr_key.shape.adapt(self._dim, value, allow_broadcast=True)
        except Exception as e:
            msg = f"Attribute '{name}' (given as '{matched.pattern}') is not properly specified. " \
                "Not a compatible type."
            raise AttributeException(msg) from e
        if value is None:
            msg = f"Attribute '{name}' (given as '{matched.pattern}') is not properly specified. " \
                "Not a compatible shape."
            raise AttributeException(msg)
        return value


class AttributeResolver(_BaseAttributeResolver):
    """
    Wraps a Database of AttributeArrays to provide the ability to access to that data
    by AttributeKey within a particular ModuleNamespace.
    """

    @overload
    def resolve(self, attr: tuple[AbsoluteName, AttributeKey[type[int]]]) -> NDArray[np.int64]:
        ...

    @overload
    def resolve(self, attr: tuple[AbsoluteName, AttributeKey[type[float]]]) -> NDArray[np.float64]:
        ...

    @overload
    def resolve(self, attr: tuple[AbsoluteName, AttributeKey[type[str]]]) -> NDArray[np.str_]:
        ...

    @overload
    def resolve(self, attr: tuple[AbsoluteName, AttributeKey[Any]]) -> NDArray[Any]:
        ...

    def resolve(self, attr: tuple[AbsoluteName, AttributeKey]) -> NDArray:
        """Retrieve the value of a specific attribute, typed and shaped appropriately."""
        return super()._resolve(attr)

    def resolve_txn_series(
        self,
        attributes: Sequence[tuple[AbsoluteName, AttributeKey]],
    ) -> Generator[Iterable[AttributeValue], None, None]:
        """
        Generates the series of values for the given attributes. Each item produced by the generator
        is a sequence of values, one for each attribute (in the given order). The sequence of items is generated
        in simulation order -- day=0, tau step=0, node=0; then day=0, tau_step=0; node=1; and so on.
        """
        days = self._dim.days
        taus = self._dim.tau_steps
        nodes = self._dim.nodes

        if any(ak.shape != Shapes.TxN for _, ak in attributes):
            msg = "Cannot generate a TxN series unless all attributes are TxN."
            raise AttributeException(msg)

        attr_values = [self.resolve(a) for a in attributes]

        for t in range(days):
            node_values = [
                [array[t, n] for array in attr_values]
                for n in range(nodes)
            ]
            for _ in range(taus):
                for vals in node_values:
                    yield vals


class NamespacedAttributeResolver(_BaseAttributeResolver):
    """
    Wraps a Database of AttributeArrays to provide the ability to access to that data
    by AttributeKey within a particular ModuleNamespace.
    """

    _namespace: ModuleNamespace

    def __init__(self, data: Database[AttributeArray], dim: SimDimensions, namespace: ModuleNamespace):
        super().__init__(data, dim)
        self._namespace = namespace

    @overload
    def resolve(self, attr: AttributeKey[type[int]]) -> NDArray[np.int64]: ...
    @overload
    def resolve(self, attr: AttributeKey[type[float]]) -> NDArray[np.float64]: ...
    @overload
    def resolve(self, attr: AttributeKey[type[str]]) -> NDArray[np.str_]: ...
    @overload
    def resolve(self, attr: AttributeKey[Any]) -> NDArray[Any]: ...

    def resolve(self, attr: AttributeKey) -> NDArray:
        """Retrieve the value of a specific attribute, typed and shaped appropriately."""
        # Assume that we've already validated the attributes, so we don't have to do that every time.
        # In practice we should not see misses or type/shape incompatibilities.
        name = self._namespace.to_absolute(attr.name)
        return super()._resolve((name, attr))

    def resolve_name(self, attr_name: str) -> NDArray:
        """
        Retrieve the value of a specific attribute by name.
        Note: using `resolve()` is preferred when possible,
        since in that case we can usually provide a properly typed result.
        """
        name = self._namespace.to_absolute(attr_name)
        matched = self._data.query(name)
        if matched is None:
            msg = f"Missing attribute '{name}'"
            raise AttributeException(msg)
        else:
            return matched.value


########################
# Simulation functions #
########################


T_co = TypeVar('T_co', bound=np.generic, covariant=True)
"""The result type of a SimulationFunction."""

_DeferredT = TypeVar('_DeferredT', bound=np.generic)
"""The result type of a SimulationFunction during deference."""


class _Context:
    def data(self, attribute: AttributeKey) -> NDArray:
        """Retrieve the value of a specific attribute."""
        raise ValueError("Invalid access of function context.")

    @property
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""
        raise ValueError("Invalid access of function context.")

    @property
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""
        raise ValueError("Invalid access of function context.")

    def defer(self, other: 'SimulationFunction[T_co]') -> NDArray[T_co]:
        """Defer processing to another similarly-typed instance of a SimulationFunction."""
        raise ValueError("Invalid access of function context.")


_EMPTY_CONTEXT = _Context()


class _RealContext(_Context):
    # The following attributes make up the evaluation context.
    # They are set for the duration of `__call__()` and cleared afterwards.
    # This allows implementations to use `self` to access the context during
    # evaluation. It also allows us to cache attribute resolution results
    # as implementations may be doing that within a hot loop.
    _cache: dict[AttributeKey, AttributeArray]
    _data: NamespacedAttributeResolver
    _dim: SimDimensions
    _rng: np.random.Generator

    def __init__(self, data: NamespacedAttributeResolver, dim: SimDimensions, rng: np.random.Generator):
        self._cache = {}
        self._data = data
        self._dim = dim
        self._rng = rng

    def data(self, attribute: AttributeKey) -> NDArray:
        """Retrieve the value of a specific attribute."""
        if (result := self._cache.get(attribute)) is None:
            result = self._data.resolve(attribute)
            self._cache[attribute] = result
        return result

    @property
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""
        return self._dim

    @property
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""
        return self._rng

    def defer(self, other: 'SimulationFunction[_DeferredT]') -> NDArray[_DeferredT]:
        """Defer processing to another similarly-typed instance of a SimulationFunction."""
        return other(self._data, self._dim, self._rng)


class SimulationFunction(ABC, Generic[T_co]):
    """
    A function which runs in the context of a simulation to produce a value (as a numpy array).
    Implement a SimulationFunction by extending this class and overriding the `evaluate()` method.
    """

    attributes: Sequence[AttributeDef] = ()
    """The attribute definitions which describe the data requirements for this function."""

    _ctx: _Context = _EMPTY_CONTEXT

    def __call__(
        self,
        data: NamespacedAttributeResolver,
        dim: SimDimensions,
        rng: np.random.Generator,
    ) -> NDArray[T_co]:
        try:
            self._ctx = _RealContext(data, dim, rng)
            return self.evaluate()
        finally:
            self._ctx = _EMPTY_CONTEXT

    @abstractmethod
    def evaluate(self) -> NDArray[T_co]:
        """
        Implement this method to provide logic for the function.
        Your implementation is free to use `data`, `dim`, and `rng` in this function body.
        You can also use `defer` to utilize another SimulationFunction instance.
        """

    @overload
    def data(self, attribute: AttributeKey[type[int]]) -> NDArray[np.int64]: ...
    @overload
    def data(self, attribute: AttributeKey[type[float]]) -> NDArray[np.float64]: ...
    @overload
    def data(self, attribute: AttributeKey[type[str]]) -> NDArray[np.str_]: ...
    @overload
    def data(self, attribute: AttributeKey[Any]) -> NDArray[Any]: ...

    def data(self, attribute: AttributeKey) -> NDArray:
        """Retrieve the value of a specific attribute."""
        if attribute not in self.attributes:
            msg = "You've accessed an attribute which you did not declare as a dependency!"
            raise ValueError(msg)
        return self._ctx.data(attribute)

    @property
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""
        return self._ctx.dim

    @property
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""
        return self._ctx.rng

    @final
    def defer(self, other: 'SimulationFunction[_DeferredT]') -> NDArray[_DeferredT]:
        """Defer processing to another similarly-typed instance of a SimulationFunction."""
        return self._ctx.defer(other)
