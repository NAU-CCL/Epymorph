"""General simulation requisites and utility functions."""
import logging
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cache, cached_property
from importlib import reload
from typing import (Any, Callable, Generator, Generic, Iterable, NamedTuple,
                    Self, Sequence, Type, TypeVar, final, overload)

import numpy as np
from jsonpickle.util import is_picklable
from numpy.random import SeedSequence
from numpy.typing import NDArray

from epymorph.data_shape import DataShape, Shapes, SimDimensions
from epymorph.data_type import (AttributeArray, AttributeType, AttributeValue,
                                dtype_as_np, dtype_check)
from epymorph.database import (AbsoluteName, AttributeName, Database,
                               ModuleNamespace)
from epymorph.error import AttributeException
from epymorph.geography.scope import GeoScope
from epymorph.util import acceptable_name, are_instances, are_unique


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


class TickIndex(NamedTuple):
    """A zero-based index of the simulation tau steps."""
    step: int  # which tau step within that day (zero-indexed)


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


T_co = TypeVar('T_co', covariant=True)
"""The result type of a SimulationFunction."""

_DeferredT = TypeVar('_DeferredT')
"""The result type of a SimulationFunction during deference."""


class _Context(ABC):
    """
    The evaluation context of a SimulationFunction. We want SimulationFunction
    instances to be able to access properties of the simulation by using
    various methods on `self`. But we also want to instantiate SimulationFunctions
    before the simulation context exists! Hence this object starts out "empty" 
    and will be swapped for a "real" context when the function is evaluated in
    a simulation context object.
    """

    @abstractmethod
    def data(self, attribute: AttributeDef) -> NDArray:
        """Retrieve the value of an attribute."""

    @property
    @abstractmethod
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""

    @property
    @abstractmethod
    def scope(self) -> GeoScope:
        """The simulation GeoScope."""

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""

    @abstractmethod
    def export(self) -> tuple[NamespacedAttributeResolver, SimDimensions, GeoScope, np.random.Generator]:
        """Tuples the contents of this context so it can be re-used (see: defer())."""


class _EmptyContext(_Context):
    def data(self, attribute: AttributeDef) -> NDArray:
        raise TypeError("Invalid access of function context.")

    @property
    def dim(self) -> SimDimensions:
        raise TypeError("Invalid access of function context.")

    @property
    def scope(self) -> GeoScope:
        raise TypeError("Invalid access of function context.")

    @property
    def rng(self) -> np.random.Generator:
        raise TypeError("Invalid access of function context.")

    def export(self) -> tuple[NamespacedAttributeResolver, SimDimensions, GeoScope, np.random.Generator]:
        raise TypeError("Invalid access of function context.")


_EMPTY_CONTEXT = _EmptyContext()


class _RealContext(_Context):
    _cached_data: Callable[[AttributeDef], AttributeArray]
    _data: NamespacedAttributeResolver
    _dim: SimDimensions
    _scope: GeoScope
    _rng: np.random.Generator

    def __init__(self, data: NamespacedAttributeResolver, dim: SimDimensions, scope: GeoScope, rng: np.random.Generator):
        self._cached_data = cache(data.resolve)
        self._data = data
        self._dim = dim
        self._scope = scope
        self._rng = rng

    def data(self, attribute: AttributeDef) -> NDArray:
        # attribute resolutions are cached because implementations may be
        # calling this function within a hot loop
        # (and we can't just throw @cache on this method because it interferes
        # with abstract method overriding)
        return self._cached_data(attribute)

    @property
    def dim(self) -> SimDimensions:
        return self._dim

    @property
    def scope(self) -> GeoScope:
        return self._scope

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def export(self) -> tuple[NamespacedAttributeResolver, SimDimensions, GeoScope, np.random.Generator]:
        return (self._data, self._dim, self._scope, self._rng)


_TypeT = TypeVar("_TypeT")


class SimulationFunctionClass(ABCMeta):
    """
    The metaclass for SimulationFunctions.
    Used to verify proper class implementation.
    """
    def __new__(
        mcs: Type[_TypeT],
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
    ) -> _TypeT:
        # Check requirements if this class overrides it.
        # (Otherwise class will inherit from parent.)
        if (reqs := dct.get("requirements")) is not None:
            if not isinstance(reqs, (list, tuple)):
                raise TypeError(
                    f"Invalid requirements in {name}: please specify as a list or tuple."
                )
            if not are_instances(reqs, AttributeDef):
                raise TypeError(
                    f"Invalid requirements in {name}: must be instances of AttributeDef."
                )
            if not are_unique(r.name for r in reqs):
                raise TypeError(
                    f"Invalid requirements in {name}: requirement names must be unique."
                )
            # Make requirements list immutable
            dct["requirements"] = tuple(reqs)

        # Check serializable
        if not is_picklable(name, mcs):
            raise TypeError(
                f"Invalid simulation function {name}: classes must be serializable (using jsonpickle)."
            )

        # NOTE: is_picklable() is misleading here; it does not guarantee that instances of a class are picklable,
        # nor (if you called it against an instance) that all of the instance's attributes are picklable.
        # jsonpickle simply ignores unpicklable fields, decoding objects into attribute swiss cheese.
        # It will be more effective to check that all of the attributes of an object are picklable before we try to
        # serialize it... Thus I don't think we can guarantee picklability at class definition time.
        # Something like:
        #   [(n, is_picklable(n, x)) for n, x in obj.__dict__.items()]
        # Why worry? Lambda functions are probably the most likely problem; they're not picklable by default.
        # But a simple workaround is to use a def function and, if needed, partial function application.

        return super().__new__(mcs, name, bases, dct)


class BaseSimulationFunction(ABC, Generic[T_co], metaclass=SimulationFunctionClass):
    """
    A function which runs in the context of a simulation to produce a value (as a numpy array).
    This base class exists to share functionality without limiting the function signature
    of evaluate().
    """

    requirements: Sequence[AttributeDef] = ()
    """The attribute definitions describing the data requirements for this function."""

    _ctx: _Context = _EMPTY_CONTEXT

    def with_context(
        self,
        data: NamespacedAttributeResolver,
        dim: SimDimensions,
        scope: GeoScope,
        rng: np.random.Generator,
    ) -> Self:
        """
        Constructs a clone of this instance which has access to the given context.
        epymorph calls this function; you generally don't need to.
        """
        # clone this instance, then run evaluate on that; accomplishes two things:
        # 1. don't have to worry about cleaning up _ctx
        # 2. instances can use @cached_property without surprising results
        clone = deepcopy(self)
        setattr(clone, "_ctx", _RealContext(data, dim, scope, rng))
        return clone

    def data(self, attribute: AttributeDef | str) -> NDArray:
        """Retrieve the value of a specific attribute."""
        if isinstance(attribute, str):
            name = attribute
            req = next((r for r in self.requirements if r.name == attribute), None)
        else:
            name = attribute.name
            req = attribute
        if req is None or req not in self.requirements:
            raise ValueError(
                f"Simulation function {self.__class__.__name__} accessed an attribute ({name}) "
                "which you did not declare as a requirement."
            )
        return self._ctx.data(req)

    @property
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""
        return self._ctx.dim

    @property
    def scope(self) -> GeoScope:
        """The simulation GeoScope."""
        return self._ctx.scope

    @property
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""
        return self._ctx.rng


class SimulationFunction(BaseSimulationFunction[T_co]):
    """
    A function which runs in the context of a simulation to produce a value (as a numpy array).
    Implement a SimulationFunction by extending this class and overriding the `evaluate()` method.
    """

    def evaluate_in_context(
        self,
        data: NamespacedAttributeResolver,
        dim: SimDimensions,
        scope: GeoScope,
        rng: np.random.Generator,
    ) -> T_co:
        """
        Evaluate this function within a context.
        epymorph calls this function; you generally don't need to.
        """
        return super()\
            .with_context(data, dim, scope, rng)\
            .evaluate()

    @abstractmethod
    def evaluate(self) -> T_co:
        """
        Implement this method to provide logic for the function.
        Your implementation is free to use `data`, `dim`, and `rng` in this function body.
        You can also use `defer` to utilize another SimulationFunction instance.
        """

    @final
    def defer(self, other: 'SimulationFunction[_DeferredT]') -> _DeferredT:
        """Defer processing to another instance of a SimulationFunction."""
        return other.evaluate_in_context(*self._ctx.export())


class SimulationTickFunction(BaseSimulationFunction[T_co]):
    """
    A function which runs in the context of a simulation to produce a sim-time-specific value (as a numpy array).
    Implement a SimulationTickFunction by extending this class and overriding the `evaluate()` method.
    """

    def evaluate_in_context(
        self,
        data: NamespacedAttributeResolver,
        dim: SimDimensions,
        scope: GeoScope,
        rng: np.random.Generator,
        tick: Tick
    ) -> T_co:
        """
        Evaluate this function within a context.
        epymorph calls this function; you generally don't need to.
        """
        return super()\
            .with_context(data, dim, scope, rng)\
            .evaluate(tick)

    @abstractmethod
    def evaluate(self, tick: Tick) -> T_co:
        """
        Implement this method to provide logic for the function.
        Your implementation is free to use `data`, `dim`, and `rng` in this function body.
        You can also use `defer` to utilize another SimulationTickFunction instance.
        """

    @final
    def defer(self, other: 'SimulationTickFunction[_DeferredT]', tick: Tick) -> _DeferredT:
        """Defer processing to another instance of a SimulationTickFunction."""
        return other.evaluate_in_context(*self._ctx.export(), tick)


###############
# Multistrata #
###############


DEFAULT_STRATA = "all"
"""The strata name used as the default, primarily for single-strata simulations."""
META_STRATA = "meta"
"""A strata for information that concerns the other strata."""


def gpm_strata(strata_name: str) -> str:
    """The strata name for a GPM in a multistrata RUME."""
    return f"gpm:{strata_name}"
