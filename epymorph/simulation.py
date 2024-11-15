"""General simulation requisites and utility functions."""

from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from datetime import date, timedelta
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Self,
    Sequence,
    Type,
    TypeVar,
    Union,
    final,
)

import numpy as np
from jsonpickle.util import is_picklable
from numpy.random import SeedSequence
from numpy.typing import NDArray
from sympy import Expr

from epymorph.data_shape import SimDimensions
from epymorph.data_type import (
    AttributeArray,
    ScalarDType,
    ScalarValue,
    StructDType,
    StructValue,
)
from epymorph.database import (
    NAMESPACE_PLACEHOLDER,
    AttributeDef,
    AttributeName,
    Database,
    DataResolver,
    ModuleNamespace,
    NamePattern,
    RecursiveValue,
    ReqTree,
    is_recursive_value,
)
from epymorph.geography.scope import GeoScope
from epymorph.util import are_instances, are_unique


def default_rng(
    seed: int | SeedSequence | None = None,
) -> Callable[[], np.random.Generator]:
    """
    Convenience constructor to create a factory function for a simulation's
    random number generator, optionally with a given seed.
    """
    return lambda: np.random.default_rng(seed)


###################
# Simulation time #
###################


class Tick(NamedTuple):
    """
    A Tick bundles related time-step information.
    For instance, each time step corresponds to a calendar day,
    a numeric day (i.e., relative to the start of the simulation),
    which tau step this corresponds to, and so on.
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
    return (
        -1
        if delta.days == -1
        else tick.sim_index - tick.step + (dim.tau_steps * delta.days) + delta.step
    )


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


########################
# Simulation functions #
########################


class _Context(ABC):
    """
    The evaluation context of a SimulationFunction. We want SimulationFunction
    instances to be able to access properties of the simulation by using
    various methods on `self`. But we also want to instantiate SimulationFunctions
    before the simulation context exists! Hence this object starts out "empty"
    and will be swapped for a "full" context when the function is evaluated in
    a simulation context object. Partial contexts exist to allow easy one-off
    evaluation of SimulationFunctions without a full RUME.
    """

    def _invalid_context(
        self,
        component: Literal["data", "dim", "scope", "rng"],
    ) -> TypeError:
        err = (
            "Missing function context during evaluation.\n"
            "Simulation function tried to access "
            f"'{component}' but this has not been provided. "
            "Call `with_context()` first, providing all context that is required "
            "by this function. Then call `evaluate()` on the returned object "
            "to compute the value."
        )
        return TypeError(err)

    @abstractmethod
    def data(self, attribute: AttributeDef) -> AttributeArray:
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

    @staticmethod
    def of(
        namespace: ModuleNamespace,
        data: DataResolver | None,
        dim: SimDimensions | None,
        scope: GeoScope | None,
        rng: np.random.Generator | None,
    ) -> "_PartialContext | _FullContext":
        if (
            namespace is None
            or data is None
            or dim is None
            or scope is None
            or rng is None
        ):
            return _PartialContext(namespace, data, dim, scope, rng)
        else:
            return _FullContext(namespace, data, dim, scope, rng)


class _PartialContext(_Context):
    _namespace: ModuleNamespace
    _data: DataResolver | None
    _dim: SimDimensions | None
    _scope: GeoScope | None
    _rng: np.random.Generator | None

    def __init__(
        self,
        namespace: ModuleNamespace,
        data: DataResolver | None,
        dim: SimDimensions | None,
        scope: GeoScope | None,
        rng: np.random.Generator | None,
    ):
        self._namespace = namespace
        self._data = data
        self._dim = dim
        self._scope = scope
        self._rng = rng

    def data(self, attribute: AttributeDef) -> NDArray:
        if self._data is None:
            raise self._invalid_context("data")
        name = self._namespace.to_absolute(attribute.name)
        return self._data.resolve(name, attribute)

    @property
    def dim(self) -> SimDimensions:
        if self._dim is None:
            raise self._invalid_context("dim")
        return self._dim

    @property
    def scope(self) -> GeoScope:
        if self._scope is None:
            raise self._invalid_context("scope")
        return self._scope

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise self._invalid_context("rng")
        return self._rng


_EMPTY_CONTEXT = _PartialContext(NAMESPACE_PLACEHOLDER, None, None, None, None)


class _FullContext(_Context):
    _namespace: ModuleNamespace
    _data: DataResolver
    _dim: SimDimensions
    _scope: GeoScope
    _rng: np.random.Generator

    def __init__(
        self,
        namespace: ModuleNamespace,
        data: DataResolver,
        dim: SimDimensions,
        scope: GeoScope,
        rng: np.random.Generator,
    ):
        self._namespace = namespace
        self._data = data
        self._dim = dim
        self._scope = scope
        self._rng = rng

    def data(self, attribute: AttributeDef) -> NDArray:
        name = self._namespace.to_absolute(attribute.name)
        return self._data.resolve(name, attribute)

    @property
    def dim(self) -> SimDimensions:
        return self._dim

    @property
    def scope(self) -> GeoScope:
        return self._scope

    @property
    def rng(self) -> np.random.Generator:
        return self._rng


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
                    f"Invalid requirements in {name}: "
                    "please specify as a list or tuple."
                )
            if not are_instances(reqs, AttributeDef):
                raise TypeError(
                    f"Invalid requirements in {name}: "
                    "must be instances of AttributeDef."
                )
            if not are_unique(r.name for r in reqs):
                raise TypeError(
                    f"Invalid requirements in {name}: "
                    "requirement names must be unique."
                )
            # Make requirements list immutable
            dct["requirements"] = tuple(reqs)

        # Check serializable
        if not is_picklable(name, mcs):
            raise TypeError(
                f"Invalid simulation function {name}: "
                "classes must be serializable (using jsonpickle)."
            )

        # NOTE: is_picklable() is misleading here; it does not guarantee that instances
        # of a class are picklable, nor (if you called it against an instance) that all
        # of the instance's attributes are picklable. jsonpickle simply ignores
        # unpicklable fields, decoding objects into attribute swiss cheese.
        # It will be more effective to check that all of the attributes of an object
        # are picklable before we try to serialize it...
        # Thus I don't think we can guarantee picklability at class definition time.
        # Something like:
        #   [(n, is_picklable(n, x)) for n, x in obj.__dict__.items()]
        # Why worry? Lambda functions are probably the most likely problem;
        # they're not picklable by default.
        # But a simple workaround is to use a def function and,
        # if needed, partial function application.

        return super().__new__(mcs, name, bases, dct)


T_co = TypeVar("T_co", covariant=True)
"""The result type of a SimulationFunction."""

_DeferResultT = TypeVar("_DeferResultT")
"""The result type of a SimulationFunction during deference."""
_DeferFunctionT = TypeVar("_DeferFunctionT", bound="BaseSimulationFunction")
"""The type of a SimulationFunction during deference."""


class BaseSimulationFunction(ABC, Generic[T_co], metaclass=SimulationFunctionClass):
    """
    A function which runs in the context of a simulation to produce a value
    (as a numpy array). This base class exists to share functionality without
    limiting the function signature of evaluate().
    """

    requirements: Sequence[AttributeDef] = ()
    """The attribute definitions describing the data requirements for this function."""

    randomized: bool = False
    """Should this function be re-evaluated every time it's referenced in a RUME?
    (Mostly useful for randomized results.) If False, even a function that utilizes
    the context RNG will only be computed once, resulting in a single random value
    that is shared by all references during evaluation."""

    _ctx: _FullContext | _PartialContext = _EMPTY_CONTEXT

    @final
    def with_context(
        self,
        namespace: ModuleNamespace = NAMESPACE_PLACEHOLDER,
        params: "Mapping[str, ParamValue] | None" = None,
        dim: SimDimensions | None = None,
        scope: GeoScope | None = None,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Constructs a clone of this instance which has access to the given context."""
        # This version allows users to specify data using strings for names.
        # epymorph should use `with_context_internal()` whenever possible.

        if params is None:
            params = {}
        try:
            for p in params:
                AttributeName(p)
        except ValueError:
            err = (
                "When evaluating a sim function this way, namespaced params "
                "are not allowed (names using '::') because those values would "
                "not be able to contribute to the evaluation. "
                "Specify param names as simple strings instead."
            )
            raise ValueError(err)
        reqs = ReqTree.of(
            {namespace.to_absolute(req.name): req for req in self.requirements},
            Database({NamePattern.parse(k): v for k, v in params.items()}),
        )
        data = reqs.evaluate(dim, scope, rng)
        return self.with_context_internal(namespace, data, dim, scope, rng)

    def with_context_internal(
        self,
        namespace: ModuleNamespace = NAMESPACE_PLACEHOLDER,
        data: DataResolver | None = None,
        dim: SimDimensions | None = None,
        scope: GeoScope | None = None,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Constructs a clone of this instance which has access to the given context."""
        # clone this instance, then run evaluate on that; accomplishes two things:
        # 1. don't have to worry about cleaning up _ctx
        # 2. instances can use @cached_property without surprising results
        clone = deepcopy(self)
        setattr(clone, "_ctx", _Context.of(namespace, data, dim, scope, rng))
        return clone

    @final
    def defer_context(
        self,
        other: _DeferFunctionT,
    ) -> _DeferFunctionT:
        """Defer processing to another instance of a SimulationFunction."""
        return other.with_context_internal(
            namespace=self._ctx._namespace,
            data=self._ctx._data,
            dim=self._ctx._dim,
            scope=self._ctx._scope,
            rng=self._ctx._rng,
        )

    @final
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
                f"Simulation function {self.__class__.__name__} "
                f"accessed an attribute ({name}) "
                "which you did not declare as a requirement."
            )
        return self._ctx.data(req)

    @final
    @property
    def dim(self) -> SimDimensions:
        """The simulation dimensions."""
        return self._ctx.dim

    @final
    @property
    def scope(self) -> GeoScope:
        """The simulation GeoScope."""
        return self._ctx.scope

    @final
    @property
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""
        return self._ctx.rng


@is_recursive_value.register
def _(value: BaseSimulationFunction) -> RecursiveValue | None:
    return RecursiveValue(value.requirements, value.randomized)


class SimulationFunction(BaseSimulationFunction[T_co]):
    """
    A function which runs in the context of a simulation to produce a value
    (as a numpy array).
    Implement a SimulationFunction by extending this class and overriding the
    `evaluate()` method.
    """

    @abstractmethod
    def evaluate(self) -> T_co:
        """
        Implement this method to provide logic for the function.
        Your implementation is free to use `data`, `dim`, and `rng`.
        You can also use `defer` to utilize another SimulationFunction instance.
        """

    @final
    def defer(self, other: "SimulationFunction[_DeferResultT]") -> _DeferResultT:
        """Defer processing to another instance of a SimulationFunction."""
        return self.defer_context(other).evaluate()


class SimulationTickFunction(BaseSimulationFunction[T_co]):
    """
    A function which runs in the context of a simulation to produce a sim-time-specific
    value (as a numpy array). Implement a SimulationTickFunction by extending this class
    and overriding the `evaluate()` method.
    """

    @abstractmethod
    def evaluate(self, tick: Tick) -> T_co:
        """
        Implement this method to provide logic for the function.
        Your implementation is free to use `data`, `dim`, and `rng`.
        You can also use `defer` to utilize another SimulationTickFunction instance.
        """

    @final
    def defer(
        self, other: "SimulationTickFunction[_DeferResultT]", tick: Tick
    ) -> _DeferResultT:
        """Defer processing to another instance of a SimulationTickFunction."""
        return self.defer_context(other).evaluate(tick)


########################
# Parameter Resolution #
########################


ListValue = Sequence[Union[ScalarValue, StructValue, "ListValue"]]
ParamValue = Union[
    ScalarValue,
    StructValue,
    ListValue,
    SimulationFunction,
    Expr,
    NDArray[ScalarDType | StructDType],
]
"""All acceptable input forms for parameter values."""


# def evaluate_param(
#     node: ReqNode[ParamValue],
#     data: DataResolver,
#     dim: SimDimensions | None,
#     scope: GeoScope | None,
#     rng: np.random.Generator | None,
# ) -> AttributeArray:
#     if isinstance(node.resolution, DefaultValue | ParameterValue):
#         raw_value = node.value
#     else:
#         # MissingValue case should have already raised an error
#         err = f"Unsupported resolution type ({type(node.resolution)})"
#         raise AttributeException(err)

#     if raw_value is None:
#         # This shouldn't happen -- when a node's resolution is
#         # Default/ParameterValue, it should always have a value.
#         # So this error indicates a bug in the construction of the ReqTree.
#         err = "Internal error: unable to resolve value for requirement."
#         raise AttributeException(err)

#     # Raw value conversions:
#     if isinstance(raw_value, Expr):
#         # Automatically convert sympy expressions into a ParamFunction instance.
#         try:
#             raw_value = ParamExpressionTimeAndNode(raw_value)
#         except ValueError as e:
#             raise AttributeException(str(e)) from None

#     # Otherwise, evaluate and store the parameter based on its type.
#     if isinstance(raw_value, SimulationFunction):
#         # SimFunc: depth-first evaluation guarantees `resolved`
#         # contains all of the data that we will need.
#         namespace = node.name.to_namespace()
#         sim_func = raw_value.with_context_internal(namespace, data, dim, scope, rng)
#         return np.asarray(sim_func.evaluate())
#     elif isinstance(raw_value, np.ndarray):
#         # numpy array: make a copy so we don't risk unexpected mutations
#         return raw_value.copy()
#     elif isinstance(raw_value, int | float | str | tuple | Sequence):
#         # scalar value or python collection: re-pack it as a numpy array
#         return np.asarray(raw_value, dtype=None)
#     elif (
#         isinstance(raw_value, type)  #
#         and issubclass(raw_value, SimulationFunction)
#     ):
#         # forgot to instantiate: a common error worth checking for
#         err = "ParamFunction/Adrio was given as a class instead of an instance."
#         raise AttributeException(err)
#     else:
#         # unsupported value!
#         err = f"Parameter not a supported type (found: {type(raw_value)})"
#         raise AttributeException(err)


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
