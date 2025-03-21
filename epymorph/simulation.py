"""General simulation requisites and utility functions."""

import functools
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
    TypeGuard,
    TypeVar,
    Union,
    final,
)

import numpy as np
from jsonpickle.util import is_picklable
from numpy.random import SeedSequence
from numpy.typing import NDArray
from sympy import Expr
from typing_extensions import override

from epymorph.attribute import (
    NAME_PLACEHOLDER,
    AbsoluteName,
    AttributeDef,
    AttributeName,
    NamePattern,
)
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import Dimensions
from epymorph.data_type import (
    AttributeArray,
    ScalarDType,
    ScalarValue,
    StructDType,
    StructValue,
)
from epymorph.database import (
    Database,
    DataResolver,
    RecursiveValue,
    ReqTree,
    is_recursive_value,
)
from epymorph.geography.scope import GeoScope
from epymorph.time import TimeFrame
from epymorph.util import are_instances, are_unique


def default_rng(
    seed: int | SeedSequence | None = None,
) -> Callable[[], np.random.Generator]:
    """
    Convenience constructor to create a factory function for a simulation's
    random number generator.

    Parameters
    ----------
    seed : int | SeedSequence, optional
        Construct a generate with this random seed. By default, the generator will be
        "unseeded", that is seeded in an unpredictable way.

    Returns
    -------
    Callable[[], np.random.Generator]
        A function that creates a random number generator.
    """
    return lambda: np.random.default_rng(seed)


###################
# Simulation time #
###################


class Tick(NamedTuple):
    """
    A Tick represents a single time-step in a simulation.
    This object bundles the related information for a single tick.
    For instance, each time step corresponds to a calendar day,
    a numeric day (i.e., relative to the start of the simulation),
    which tau step this corresponds to, and so on.

    Tick is a NamedTuple.
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
    """
    An index identifying tau steps within a day.
    For example, if a RUME's movement model declares three tau steps,
    the first six tick indices will be 0, 1, 2, 0, 1, 2.
    Indices 3 or more would be invalid for this RUME.

    TickIndex is a NamedTuple.
    """

    step: int
    """The index of the tau step."""


class TickDelta(NamedTuple):
    """
    An offset relative to a Tick expressed as a number of days which should elapse,
    and the step on which to end up. In applying this delta, it does not matter which
    step we start on.

    See Also
    --------
    [`resolve_tick_delta`](`epymorph.simulation.resolve_tick_delta`) combines an
    absolute tick with a relative delta to get another absolute tick.
    """

    days: int
    """The number of whole days."""
    step: int
    """Which tau step within that day (zero-indexed)."""


NEVER = TickDelta(-1, -1)
"""
A special TickDelta which expresses an event that should never happen.
Any Tick plus Never returns Never.
"""


def resolve_tick_delta(tau_steps_per_day: int, tick: Tick, delta: TickDelta) -> int:
    """
    Add a delta to a tick to get the index of the resulting tick.

    Parameters
    ----------
    tau_steps_per_day : int
        The number of simulation tau steps per day.
    tick : Tick
        The tick to start from.
    delta : TickDelta
        A delta to add to `tick`.

    Returns
    -------
    int
        The result of adding `delta` to `tick`, as a tick index.
    """
    return (
        -1
        if delta.days == -1
        else tick.sim_index - tick.step + (tau_steps_per_day * delta.days) + delta.step
    )


def simulation_clock(
    time_frame: TimeFrame,
    tau_step_lengths: list[float],
) -> Iterable[Tick]:
    """
    Generates the sequence of ticks which makes up the simulation clock.

    Parameters
    ----------
    time_frame : TimeFrame
        A simulation's time frame.
    tau_step_lengths : list[float]
        The simulation tau steps as fractions of one day.

    Returns
    -------
    Iterable[Tick]
        The sequence of ticks as determined by the simulation's start date,
        duration, and tau steps configuration.
    """
    one_day = timedelta(days=1)
    tau_steps = list(enumerate(tau_step_lengths))
    curr_index = 0
    curr_date = time_frame.start_date
    for day in range(time_frame.days):
        for step, tau in tau_steps:
            yield Tick(curr_index, day, curr_date, step, tau)
            curr_index += 1
        curr_date += one_day


##############################
# Simulation parameter types #
##############################


ListValue = Sequence[Union[ScalarValue, StructValue, "ListValue"]]
ParamValue = Union[
    ScalarValue,
    StructValue,
    ListValue,
    "SimulationFunction",
    Expr,
    NDArray[ScalarDType | StructDType],
]
"""
A type alias which describes all acceptable input forms for parameter values:

- scalars (according to [`ScalarValue`](`epymorph.data_type.ScalarValue`))
- tuples (single structured values, according to
    [`StructValue`](`epymorph.data_type.StructValue`))
- Python lists of scalars or tuples, which may be nested
- Numpy arrays
- [`SimulationFunctions`](`epymorph.simulation.SimulationFunction`)
- [`sympy expressions`](`sympy.core.expr.Expr`)
"""


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
        component: Literal["data", "scope", "time_frame", "ipm", "rng"],
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

    @property
    @abstractmethod
    def name(self) -> AbsoluteName:
        """The name under which this attribute is being evaluated."""

    @abstractmethod
    def data(self, attribute: AttributeDef) -> AttributeArray:
        """Retrieve the value of an attribute."""

    @property
    @abstractmethod
    def scope(self) -> GeoScope:
        """The simulation GeoScope."""

    @property
    @abstractmethod
    def time_frame(self) -> TimeFrame:
        """The simulation time frame."""

    @property
    @abstractmethod
    def ipm(self) -> BaseCompartmentModel:
        """The simulation's IPM."""

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""

    @property
    @abstractmethod
    def dim(self) -> Dimensions:
        """Simulation dimensions."""

    @staticmethod
    def of(
        name: AbsoluteName,
        data: DataResolver | None,
        scope: GeoScope | None,
        time_frame: TimeFrame | None,
        ipm: BaseCompartmentModel | None,
        rng: np.random.Generator | None,
    ) -> "_PartialContext | _FullContext":
        if (
            name is None
            or data is None
            or scope is None
            or time_frame is None
            or ipm is None
            or rng is None
        ):
            return _PartialContext(name, data, scope, time_frame, ipm, rng)
        else:
            return _FullContext(name, data, scope, time_frame, ipm, rng)


class _PartialContext(_Context):
    _name: AbsoluteName
    _data: DataResolver | None
    _scope: GeoScope | None
    _time_frame: TimeFrame | None
    _ipm: BaseCompartmentModel | None
    _rng: np.random.Generator | None
    _dim: Dimensions

    def __init__(
        self,
        name: AbsoluteName,
        data: DataResolver | None,
        scope: GeoScope | None,
        time_frame: TimeFrame | None,
        ipm: BaseCompartmentModel | None,
        rng: np.random.Generator | None,
    ):
        self._name = name
        self._data = data
        self._scope = scope
        self._time_frame = time_frame
        self._ipm = ipm
        self._rng = rng
        self._dim = Dimensions.of(
            T=time_frame.duration_days if time_frame is not None else None,
            N=scope.nodes if scope is not None else None,
            C=ipm.num_compartments if ipm is not None else None,
            E=ipm.num_events if ipm is not None else None,
        )

    @property
    def name(self) -> AbsoluteName:
        return self._name

    @override
    def data(self, attribute: AttributeDef) -> NDArray:
        if self._data is None:
            raise self._invalid_context("data")
        name = self._name.to_namespace().to_absolute(attribute.name)
        return self._data.resolve(name, attribute)

    @property
    @override
    def time_frame(self) -> TimeFrame:
        if self._time_frame is None:
            raise self._invalid_context("time_frame")
        return self._time_frame

    @property
    @override
    def ipm(self) -> BaseCompartmentModel:
        if self._ipm is None:
            raise self._invalid_context("ipm")
        return self._ipm

    @property
    @override
    def scope(self) -> GeoScope:
        if self._scope is None:
            raise self._invalid_context("scope")
        return self._scope

    @property
    @override
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise self._invalid_context("rng")
        return self._rng

    @property
    @override
    def dim(self) -> Dimensions:
        return self._dim


_EMPTY_CONTEXT = _PartialContext(NAME_PLACEHOLDER, None, None, None, None, None)


class _FullContext(_Context):
    _name: AbsoluteName
    _data: DataResolver
    _scope: GeoScope
    _time_frame: TimeFrame
    _ipm: BaseCompartmentModel
    _rng: np.random.Generator
    _dim: Dimensions

    def __init__(
        self,
        name: AbsoluteName,
        data: DataResolver,
        scope: GeoScope,
        time_frame: TimeFrame,
        ipm: BaseCompartmentModel,
        rng: np.random.Generator,
    ):
        self._name = name
        self._data = data
        self._scope = scope
        self._time_frame = time_frame
        self._ipm = ipm
        self._rng = rng
        self._dim = Dimensions.of(
            T=time_frame.duration_days,
            N=scope.nodes,
            C=ipm.num_compartments,
            E=ipm.num_events,
        )

    @property
    def name(self) -> AbsoluteName:
        return self._name

    @override
    def data(self, attribute: AttributeDef) -> NDArray:
        name = self._name.to_namespace().to_absolute(attribute.name)
        return self._data.resolve(name, attribute)

    @property
    @override
    def scope(self) -> GeoScope:
        return self._scope

    @property
    @override
    def time_frame(self) -> TimeFrame:
        return self._time_frame

    @property
    @override
    def ipm(self) -> BaseCompartmentModel:
        return self._ipm

    @property
    @override
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    @override
    def dim(self) -> Dimensions:
        return self._dim


_TypeT = TypeVar("_TypeT")


class SimulationFunctionClass(ABCMeta):
    """
    The metaclass for SimulationFunctions.
    Used to verify proper class implementation.
    """

    def __new__(
        cls: Type[_TypeT],
        name: str,
        bases: tuple[type, ...],
        dct: dict[str, Any],
    ) -> _TypeT:
        # Check requirements if this class overrides it.
        # (Otherwise class will inherit from parent.)
        if (reqs := dct.get("requirements")) is not None:
            # The user may specify requirements as a property, in which case we
            # can't validate much about the implementation.
            if not isinstance(reqs, property):
                # But if it's a static value, check types:
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
        if not is_picklable(name, cls):
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
        #   [(n, is_picklable(n, x)) for n, x in obj.__dict__.items()]  # noqa: ERA001
        # Why worry? Lambda functions are probably the most likely problem;
        # they're not picklable by default.
        # But a simple workaround is to use a def function and,
        # if needed, partial function application.

        if (orig_evaluate := dct.get("evaluate")) is not None:

            @functools.wraps(orig_evaluate)
            def evaluate(self, *args, **kwargs):
                result = orig_evaluate(self, *args, **kwargs)
                self.validate(result)
                return result

            dct["evaluate"] = evaluate

        return super().__new__(cls, name, bases, dct)


ResultT = TypeVar("ResultT")
"""The result type of a SimulationFunction."""

_DeferResultT = TypeVar("_DeferResultT")
"""The result type of a SimulationFunction during deference."""
_DeferFunctionT = TypeVar("_DeferFunctionT", bound="BaseSimulationFunction")
"""The type of a SimulationFunction during deference."""


class BaseSimulationFunction(ABC, Generic[ResultT], metaclass=SimulationFunctionClass):
    """
    A function which runs in the context of a simulation to produce a value
    (as a numpy array). This base class exists to share functionality without
    limiting the function signature of evaluate().

    Instances may access the context in which they are being evaluated using
    attributes and methods present on "self":
    `name`, `data`, `scope`, `time_frame`, `ipm`, `rng`, and `dim`, and may use
    methods like `defer` to pass their context on to another function for
    direct evaluation.

    BaseSimulationFunction is an abstract class.

    See Also
    --------
    Refer to
    [`SimulationFunction`](`epymorph.simulation.SimulationFunction`) and
    [`SimulationTickFunction`](`epymorph.simulation.SimulationTickFunction`)
    for more concrete subclasses.
    """

    requirements: Sequence[AttributeDef] | property = ()
    """The attribute definitions describing the data requirements for this function.

    For advanced use-cases, you may specify requirements as a property if you need it
    to be dynamically computed.
    """

    randomized: bool = False
    """Should this function be re-evaluated every time it's referenced in a RUME?
    (Mostly useful for randomized results.) If False, even a function that utilizes
    the context RNG will only be computed once, resulting in a single random value
    that is shared by all references during evaluation."""

    _ctx: _FullContext | _PartialContext = _EMPTY_CONTEXT

    @property
    def class_name(self) -> str:
        """The class name of the SimulationFunction."""
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def validate(self, result: ResultT) -> None:
        """
        Override this method to validate the function evaluation result.
        Implementations should raise an appropriate error if results
        are not valid.

        Parameters
        ----------
        result : ResultT
            The result produced from function evaluation.
        """

    @final
    def with_context(
        self,
        name: AbsoluteName = NAME_PLACEHOLDER,
        params: Mapping[str, ParamValue] | None = None,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
        ipm: BaseCompartmentModel | None = None,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """
        Constructs a clone of this instance which has access to the given context.

        All elements of the context are optional, allowing users who wish to
        quickly evaluate a function in a one-off situation to provide only the
        partial context that is strictly necessary. (It's very tedious to create
        fake context when it isn't strictly needed.) For example, a function
        might be able to calculate a result knowing only the geo scope and time frame.
        If the function tries to use a part of the context that hasn't been provided,
        an exception will be raised.

        During normal function evaluation, such as when running a simulation,
        the full context is always provided and available.

        Parameters
        ----------
        name : AbsoluteName, default=NAME_PLACEHOLDER
            The name used for the value this function produces, according
            to the evaluation context. Defaults to a generic placeholder
            name.
        params : Mapping[str, ParamValue], optional
            Additional parameter values we may need to evaluate this function.
        scope : GeoScope, optional
            The geo scope.
        time_frame : TimeFrame, optional
            The time frame.
        ipm : BaseCompartmentModel, optional
            The IPM.
        rng : np.random.Generator, optional
            A random number generator instance.
        """
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
            {name.with_id(req.name): req for req in self.requirements},
            Database({NamePattern.parse(k): v for k, v in params.items()}),
        )
        data = reqs.evaluate(scope, time_frame, ipm, rng)
        return self.with_context_internal(name, data, scope, time_frame, ipm, rng)

    def with_context_internal(
        self,
        name: AbsoluteName = NAME_PLACEHOLDER,
        data: DataResolver | None = None,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
        ipm: BaseCompartmentModel | None = None,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """
        Constructs a clone of this instance which has access to the given context.

        This method is intended for usage internal to epymorph's systems.
        Typical usage will use `with_context` instead.
        """
        # clone this instance, then run evaluate on that; accomplishes two things:
        # 1. don't have to worry about cleaning up _ctx
        # 2. instances can use @cached_property without surprising results
        clone = deepcopy(self)
        setattr(clone, "_ctx", _Context.of(name, data, scope, time_frame, ipm, rng))
        return clone

    @final
    def defer_context(
        self,
        other: _DeferFunctionT,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
    ) -> _DeferFunctionT:
        """
        Defer processing to another instance of a SimulationFunction.

        This method is intended for usage internal to epymorph's system.
        Typical usage will use `defer` instead.
        """
        return other.with_context_internal(
            name=self._ctx._name,
            data=self._ctx._data,
            scope=scope or self._ctx._scope,
            time_frame=time_frame or self._ctx._time_frame,
            ipm=self._ctx._ipm,
            rng=self._ctx._rng,
        )

    @final
    @property
    def name(self) -> AbsoluteName:
        """The name under which this attribute is being evaluated."""
        return self._ctx.name

    @final
    def data(self, attribute: AttributeDef | str) -> NDArray:
        """
        Retrieve the value of a requirement. You must declare
        the attribute in this function's `requirements` list.

        Parameters
        ----------
        attribute : AttributeDef | str
            The attribute to get, identified either by its name (string) or
            its definition object.

        Returns
        -------
        NDArray
            The attribute value.

        Raises
        ------
        ValueError
            If the attribute is not in the function's requirements
            declaration.
        """
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
    def scope(self) -> GeoScope:
        """The simulation GeoScope."""
        return self._ctx.scope

    @final
    @property
    def time_frame(self) -> TimeFrame:
        """The simulation TimeFrame."""
        return self._ctx.time_frame

    @final
    @property
    def ipm(self) -> BaseCompartmentModel:
        """The simulation IPM."""
        return self._ctx.ipm

    @final
    @property
    def rng(self) -> np.random.Generator:
        """The simulation's random number generator."""
        return self._ctx.rng

    @final
    @property
    def dim(self) -> Dimensions:
        """
        The simulation's dimensional information.
        This is a re-packaging of information contained
        in other context elements, like the geo scope.
        """
        return self._ctx.dim


@is_recursive_value.register
def _(value: BaseSimulationFunction) -> TypeGuard[RecursiveValue]:
    return True


class SimulationFunction(BaseSimulationFunction[ResultT]):
    """
    A function which runs in the context of a RUME to produce a value
    (as a numpy array). The value must be independent of the simulation state,
    and they will often be evaluated before the simulation starts.

    SimulationFunction is an abstract class. In typical usage you will not
    implement a SimulationFunction directly, but rather one of its
    more-specific child classes.

    SimulationFunction is generic in the type of result it produces (`ResultT`).

    See Also
    --------
    Notable child classes of SimulationFunction include
    [`Initializers`](`epymorph.initializer.Initializer`),
    [`ADRIOs`](`epymorph.adrio.adrio.ADRIO`), and
    [`ParamFunctions`](`epymorph.params.ParamFunction`).
    """

    @abstractmethod
    def evaluate(self) -> ResultT:
        """
        Implement this method to provide logic for the function.
        Use self methods and properties to access the simulation context or defer
        processing to another function.

        evaluate is an abstract method.

        Returns
        -------
        ResultT
            The result value.
        """

    @final
    def defer(
        self,
        other: "SimulationFunction[_DeferResultT]",
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
    ) -> _DeferResultT:
        """Defer processing to another instance of a SimulationFunction, returning
        the result of evaluation.

        This function is generic in the type of result returned by the function
        to which we are deferring.

        Parameters
        ----------
        other : SimulationFunction
            the other function to defer to
        scope : GeoScope, optional
            override the geo scope for evaluation; if None, use the same scope
        time_frame : TimeFrame, optional
            override the time frame for evaluation; if None, use the same time frame
        """
        return self.defer_context(other, scope, time_frame).evaluate()


class SimulationTickFunction(BaseSimulationFunction[ResultT]):
    """
    A function which runs in the context of a RUME to produce a value
    (as a numpy array) which is expected to vary over the run of a simulation.

    SimulationTickFunction is an abstract class. In typical usage you will not
    implement a SimulationTickFunction directly, but rather one of its
    more-specific child classes.

    SimulationTickFunction is generic in the type of result it produces (`ResultT`).

    See Also
    --------
    The only notable child class of SimulationTickFunction is
    the movement model [`MovementClause`](`epymorph.movement_model.MovementClause`).
    """

    @abstractmethod
    def evaluate(self, tick: Tick) -> ResultT:
        """
        Implement this method to provide logic for the function.
        Use self methods and properties to access the simulation context or defer
        processing to another function.

        evaluate is an abstract method.

        Parameters
        ----------
        tick : Tick
            The simulation tick being evaluated.

        Returns
        -------
        ResultT
            The result value.
        """

    @final
    def defer(
        self,
        other: "SimulationTickFunction[_DeferResultT]",
        tick: Tick,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
    ) -> _DeferResultT:
        """Defer processing to another instance of a SimulationTickFunction, returning
        the result of evaluation.

        Parameters
        ----------
        other : SimulationFunction
            the other function to defer to
        scope : GeoScope, optional
            override the geo scope for evaluation; if None, use the same scope
        time_frame : TimeFrame, optional
            override the time frame for evaluation; if None, use the same time frame
        """
        return self.defer_context(other, scope, time_frame).evaluate(tick)
