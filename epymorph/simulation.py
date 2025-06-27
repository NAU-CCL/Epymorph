"""General simulation requisites and utility functions."""

import functools
import re
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from datetime import date, timedelta
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Never,
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

from epymorph.attribute import (
    NAME_PLACEHOLDER,
    AbsoluteName,
    AttributeDef,
    AttributeName,
    NamePattern,
)
from epymorph.compartment_model import BaseCompartmentModel
from epymorph.data_shape import DataShape, Dimensions
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
from epymorph.error import MissingContextError
from epymorph.geography.scope import GeoScope
from epymorph.time import TimeFrame
from epymorph.util import are_instances, are_unique


def default_rng(
    seed: int | SeedSequence | None = None,
) -> Callable[[], np.random.Generator]:
    """
    Construct to create a factory function for a simulation's
    random number generator.

    Parameters
    ----------
    seed :
        Construct a generate with this random seed. By default, the generator will be
        "unseeded", that is seeded in an unpredictable way.

    Returns
    -------
    :
        A function that creates a random number generator.
    """
    return lambda: np.random.default_rng(seed)


###################
# Simulation time #
###################


class Tick(NamedTuple):
    """
    Represents a single time-step in a simulation, by analogy to the ticking of a clock.
    This tuple bundles the related information for a single tick. For instance, each
    time step corresponds to a calendar day, a numeric day (i.e., relative to the start
    of the simulation), which tau step this corresponds to, and so on.
    """

    sim_index: int
    """Which simulation step are we on? (0,1,2,3,...)"""
    day: int
    """Which day increment are we on? Same for each tau step: (0,0,1,1,2,2,...)"""
    date: date
    """The calendar date corresponding to `day`"""
    step: int
    """Which tau step are we on, by index? (0,1,0,1,0,1,...)"""
    tau: float
    """What's the tau length of the current step? (0.666,0.333,0.666,0.333,...)"""


class TickIndex(NamedTuple):
    """
    An index identifying tau steps within a day.
    For example, if a RUME's movement model declares three tau steps,
    the first six tick indices will be 0, 1, 2, 0, 1, 2.
    Indices 3 or more would be invalid for this RUME.
    """

    step: int
    """The index of the tau step."""


class TickDelta(NamedTuple):
    """
    An offset relative to a tick, expressed as a number of whole days which should
    elapse and the step on which to end up. In applying this delta, it does not matter
    which step we start on.

    See Also
    --------
    [epymorph.simulation.resolve_tick_delta][], the function which combines an
    absolute tick with a relative delta to get another absolute tick.
    """

    days: int
    """The number of whole days."""
    step: int
    """Which tau step within that day, by index."""


NEVER = TickDelta(-1, -1)
"""
A special `TickDelta` which expresses an event that should never happen.
Any `Tick` plus `NEVER` equals `NEVER`.
"""


def resolve_tick_delta(tau_steps_per_day: int, tick: Tick, delta: TickDelta) -> int:
    """
    Add a delta to a tick to get the index of the resulting tick.

    Parameters
    ----------
    tau_steps_per_day :
        The number of simulation tau steps per day.
    tick :
        The tick to start from.
    delta :
        A delta to add to `tick`.

    Returns
    -------
    :
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
    Generate the sequence of ticks which makes up the simulation clock.

    Parameters
    ----------
    time_frame :
        A simulation's time frame.
    tau_step_lengths :
        The simulation tau steps as fractions of one day.

    Returns
    -------
    :
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
"""
A type alias which describes acceptable input forms for parameter values which are
lists. (Necessary because this type is recursive: lists can be nested inside lists.)
"""

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

- scalars (according to [epymorph.data_type.ScalarValue][])
- tuples (single structured values, according to
    [epymorph.data_type.StructValue][])
- Python lists of scalars or tuples, which may be nested
- [epymorph.simulation.SimulationFunction][] instances
- Numpy arrays
- sympy expressions
"""


########################
# Simulation functions #
########################


class Context(ABC):
    """
    The evaluation context of a `SimulationFunction`.

    See Also
    --------
    This class is abstract so you won't instantiate it as a normal class;
    instead you may use static method [epymorph.simulation.Context.of][].
    """

    # NOTE: We want SimulationFunction instances to be able to access properties of the
    # simulation by using various methods on `self`. But we also want to instantiate
    # SimulationFunctions before the simulation context exists! Hence this object
    # starts out "empty" and will be swapped for a "full" context when the function
    # is evaluated in a simulation context object. Partial contexts exist to allow easy
    # one-off evaluation of SimulationFunctions without a full RUME.

    _args: dict[str, Any]
    """Remember the arguments that were used to construct this context instance."""

    @property
    @abstractmethod
    def name(self) -> AbsoluteName:
        """The name under which this attribute is being evaluated."""

    @abstractmethod
    def data(self, attribute: AttributeDef) -> AttributeArray:
        """
        Retrieve the value of an attribute.

        Parameters
        ----------
        attribute :
            The attribute to retrieve.

        Returns
        -------
        :
            The attribute's value.
        """

    @property
    @abstractmethod
    def scope(self) -> GeoScope:
        """The simulation `GeoScope`."""

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

    @final
    def replace(
        self,
        name: AbsoluteName | None = None,
        data: DataResolver | None = None,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
        ipm: BaseCompartmentModel | None = None,
        rng: np.random.Generator | None = None,
    ) -> "Context":
        """
        Create a new context by overriding some or all of this context's values.

        Parameters
        ----------
        name :
            The name in which a `SimulationFunction` is being evaluated.
        data :
            Data attributes which may be required by the function.
        scope :
            The geo scope.
        time_frame :
            The simulation time frame.
        ipm :
            The IPM.
        rng :
            A random number generator to use.

        Returns
        -------
        :
            The new context.
        """
        arg = self._args
        return Context.of(
            name=name or arg["name"],
            data=data or arg["data"],
            scope=scope or arg["scope"],
            time_frame=time_frame or arg["time_frame"],
            ipm=ipm or arg["ipm"],
            rng=rng or arg["rng"],
        )

    @final
    def hash(self, requirements: Sequence[AttributeDef]) -> int:
        """
        Compute a hash for the context, assuming the given set of requirements.
        This is used to quickly identify if the context has changed between evaluations.

        Parameters
        ----------
        requirements :
            The subset of requirements that are important to include when computing the
            hash. This allows the hash to ignore changes to attributes which aren't
            critical to a particular use-case.

        Returns
        -------
        :
            The hash value.
        """
        name = self.name

        if (data := self._args.get("data")) is not None:
            req_hashes = (
                data.resolve(name.with_id(req.name), req).tobytes()
                for req in requirements
            )
        else:
            req_hashes = (None,)

        if (ipm := self._args.get("ipm")) is not None:
            C = ipm.num_compartments
            E = ipm.num_events
            ipm_hash = (ipm.__class__.__name__, C, E)
        else:
            ipm_hash = None

        scope = self._args.get("scope")
        time_frame = self._args.get("time_frame")

        # Note that `None` is a hashable value.
        hash_values = tuple([str(name), scope, time_frame, ipm_hash, *req_hashes])
        return hash(hash_values)

    @staticmethod
    def of(
        name: AbsoluteName = NAME_PLACEHOLDER,
        data: DataResolver | None = None,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
        ipm: BaseCompartmentModel | None = None,
        rng: np.random.Generator | None = None,
    ) -> "Context":
        """
        Create a new context instance, which may be a partial or complete context.

        Parameters
        ----------
        name :
            The name in which a `SimulationFunction` is being evaluated.
        data :
            Data attributes which may be required by the function.
        scope :
            The geo scope.
        time_frame :
            The simulation time frame.
        ipm :
            The IPM.
        rng :
            A random number generator to use.

        Returns
        -------
        :
            The new context.
        """

        # NOTE: this function dynamically creates a concrete version of Context
        # which is an abstract class. This allows us to handle partial context in an
        # efficient manner. Rather than doing an if-check on every attribute access,
        # we can just construct properties and methods which already know if they have
        # a value or must raise a MissingContextError if accessed.

        def make_missing_context(component: str):
            def missing_context(self, *args, **kwargs) -> Never:
                err = (
                    f"Missing function context '{component}' during evaluation.\n"
                    f"The simulation function tried to access '{component}' but this "
                    "has not been provided. Call `with_context()` first, providing all "
                    "context that is required by this function. Then call `evaluate()` "
                    "on the returned object to compute the value."
                )
                raise MissingContextError(err)

            return property(missing_context)

        def make_getter(component, value):
            if value is None:
                return make_missing_context(component)
            return property(lambda self: value)

        def make_data_getter():
            if data is None:
                return make_missing_context("data")

            def data_getter(self, attribute: AttributeDef) -> AttributeArray:
                n = name.to_namespace().to_absolute(attribute.name)
                return data.resolve(n, attribute)  # type: ignore

            return data_getter

        dim = Dimensions.of(
            T=time_frame.duration_days if time_frame is not None else None,
            N=scope.nodes if scope is not None else None,
            C=ipm.num_compartments if ipm is not None else None,
            E=ipm.num_events if ipm is not None else None,
        )

        args = {
            "name": name,
            "data": data,
            "scope": scope,
            "time_frame": time_frame,
            "ipm": ipm,
            "rng": rng,
        }

        instance = type(
            "ContextInstance",
            (Context,),
            {
                "_args": args,
                "name": make_getter("name", name),
                "scope": make_getter("scope", scope),
                "time_frame": make_getter("time_frame", time_frame),
                "ipm": make_getter("ipm", ipm),
                "rng": make_getter("rng", rng),
                "dim": make_getter("dim", dim),
                "data": make_data_getter(),
            },
        )
        return instance()


def validate_context_for_shape(context: Context, shape: DataShape) -> None:
    """
    Check that the elements of a context which are required to compute the given
    shape have been provided in the context. For example, if the shape contains an "N"
    axis, the context must include a scope or else there's no knowing how long that axis
    should be.

    Parameters
    ----------
    context :
        The simulation context.
    shape :
        The shape to check.

    Raises
    ------
    MissingContextError
        If any required context has not been provided.
    """

    # NOTE: this check ensures this function remains in sync with all defined shapes.
    # We could easily add another axis designation to DataShape but forget to update
    # this function.
    shape_str = str(shape)
    if re.search("[^TNCEAx]", shape_str) is not None:
        err = (
            f"Unsupported character in shape designation: {shape_str}"
            "This indicates that the `validate_context_for_shape` function was not "
            "correctly updated when new shape designations were added."
        )
        raise RuntimeError(err)

    # Check for required context.
    if "N" in shape_str:
        context.scope  # scope is required
    if "T" in shape_str:
        context.time_frame  # time_frame is required
    if "C" in shape_str or "E" in shape_str:
        context.ipm  # ipm is required


_EMPTY_CONTEXT = Context.of(NAME_PLACEHOLDER)

_TypeT = TypeVar("_TypeT")


class SimulationFunctionClass(ABCMeta):
    """`SimulationFunction` metaclass; enforces proper implementation."""

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
"""The result type of a `SimulationFunction`."""

DeferResultT = TypeVar("DeferResultT")
"""The result type of a `SimulationFunction` during deference."""
DeferFunctionT = TypeVar("DeferFunctionT", bound="BaseSimulationFunction")
"""The type of a `SimulationFunction` during deference."""


class BaseSimulationFunction(ABC, Generic[ResultT], metaclass=SimulationFunctionClass):
    """
    A function which runs in the context of a simulation to produce a value
    (as a numpy array).

    Instances may access the context in which they are being evaluated using
    attributes and methods present on "self":
    `name`, `data`, `scope`, `time_frame`, `ipm`, `rng`, and `dim`, and may use
    methods like `defer` to pass their context on to another function for
    direct evaluation.

    This class is generic on the type of result it produces (`ResultT`).


    See Also
    --------
    Refer to
    [epymorph.simulation.SimulationFunction][] and
    [epymorph.simulation.SimulationTickFunction][] for more concrete subclasses.
    """

    # NOTE: this base class exists so that we don't limit the signature of `evaluate`.
    # `SimulationTickFunction` evaluates using the current tick, while
    # `SimulationFunction` does not require any parameters.

    requirements: Sequence[AttributeDef] | property = ()
    """
    The attribute definitions describing the data requirements for this function.

    For advanced use-cases, you may specify requirements as a property if you need it
    to be dynamically computed.
    """

    randomized: bool = False
    """Should this function be re-evaluated every time it's referenced in a RUME?
    (Mostly useful for randomized results.) If False, even a function that utilizes
    the context RNG will only be computed once, resulting in a single random value
    that is shared by all references during evaluation."""

    _ctx: Context = _EMPTY_CONTEXT

    @property
    def class_name(self) -> str:
        """The class name of the `SimulationFunction`."""
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    def validate(self, result: ResultT) -> None:
        """
        Override this method to validate the function evaluation result.
        If not overridden, the default is to do no validation.
        Implementations should raise an appropriate error if results
        are not valid.

        Parameters
        ----------
        result :
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
        Construct a clone of this instance which has access to the given context.

        All elements of the context are optional, allowing users who wish to
        quickly evaluate a function in a one-off situation to provide only the
        partial context that is strictly necessary. (It's very tedious to create
        fake context when it isn't strictly needed.) For example, a function
        might be able to calculate a result knowing only the geo scope and time frame.
        During normal function evaluation, such as when running a simulation,
        the full context is always provided and available.

        Parameters
        ----------
        name :
            The name used for the value this function produces, according
            to the evaluation context. Defaults to a generic placeholder
            name.
        params :
            Additional parameter values we may need to evaluate this function.
        scope :
            The geo scope.
        time_frame :
            The time frame.
        ipm :
            The IPM.
        rng :
            A random number generator instance.

        Returns
        -------
        :
            A clone with new context information.

        Raises
        ------
        MissingContextError
            If the function tries to use a part of the context that hasn't been
            provided.
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
        ctx = Context.of(name, data, scope, time_frame, ipm, rng)
        return self.with_context_internal(ctx)

    def with_context_internal(self, context: Context) -> Self:
        """
        Construct a clone of this instance which has access to the given context.

        This method is intended for usage internal to epymorph's systems.
        Typical usage will use `with_context` instead.
        """
        # clone this instance, then run evaluate on that; accomplishes two things:
        # 1. don't have to worry about cleaning up _ctx
        # 2. instances can use @cached_property without surprising results
        clone = deepcopy(self)
        setattr(clone, "_ctx", context)
        return clone

    @final
    def defer_context(
        self,
        other: DeferFunctionT,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
    ) -> DeferFunctionT:
        """
        Defer processing to another instance of a `SimulationFunction`.

        This method is intended for usage internal to epymorph's system.
        Typical usage will use `defer` instead.
        """
        new_ctx = self._ctx.replace(
            scope=scope,
            time_frame=time_frame,
        )
        return other.with_context_internal(new_ctx)

    @final
    @property
    def name(self) -> AbsoluteName:
        """The name under which this attribute is being evaluated."""
        return self._ctx.name

    @final
    def data(self, attribute: AttributeDef | str) -> NDArray:
        """
        Retrieve the value of a requirement. You must declare the attribute in this
        function's `requirements` list.

        Parameters
        ----------
        attribute :
            The attribute to get, identified either by its name (string) or its
            definition object.

        Returns
        -------
        :
            The attribute value.

        Raises
        ------
        ValueError
            If the attribute is not in the function's requirements declaration.
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
    def context(self) -> Context:
        """The full context object."""
        return self._ctx

    @final
    @property
    def scope(self) -> GeoScope:
        """The simulation geo scope."""
        return self._ctx.scope

    @final
    @property
    def time_frame(self) -> TimeFrame:
        """The simulation time frame."""
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

    `SimulationFunction` is an abstract class. In typical usage you will not
    implement a `SimulationFunction` directly, but rather one of its
    more-specific child classes.

    `SimulationFunction` is generic in the type of result it produces (`ResultT`).

    See Also
    --------
    Notable child classes of SimulationFunction include
    [epymorph.initializer.Initializer][],
    [epymorph.adrio.adrio.ADRIO][], and
    [epymorph.params.ParamFunction][].
    """

    @abstractmethod
    def evaluate(self) -> ResultT:
        """
        Implement this method to provide logic for the function.
        Use self methods and properties to access the simulation context or defer
        processing to another function.

        Returns
        -------
        :
            The result value.
        """

    @final
    def defer(
        self,
        other: "SimulationFunction[DeferResultT]",
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
    ) -> DeferResultT:
        """
        Defer processing to another instance of a `SimulationFunction`, returning
        the result of evaluation.

        This function is generic in the type of result returned by the function
        to which we are deferring (`DeferResultT`).

        Parameters
        ----------
        other :
            The other function to defer to.
        scope :
            Override the geo scope for evaluation; if None, use the same scope.
        time_frame :
            Override the time frame for evaluation; if None, use the same time frame.

        Returns
        -------
        :
            The result value.
        """
        return self.defer_context(other, scope, time_frame).evaluate()


class SimulationTickFunction(BaseSimulationFunction[ResultT]):
    """
    A function which runs in the context of a RUME to produce a value
    (as a numpy array) which is expected to vary over the run of a simulation.

    In typical usage you will not implement a `SimulationTickFunction` directly,
    but rather one of its more-specific child classes.

    `SimulationTickFunction` is generic in the type of result it produces (`ResultT`).

    See Also
    --------
    The only notable child class is [epymorph.movement_model.MovementClause][].
    """

    @abstractmethod
    def evaluate(self, tick: Tick) -> ResultT:
        """
        Implement this method to provide logic for the function.
        Use self methods and properties to access the simulation context or defer
        processing to another function.

        Parameters
        ----------
        tick :
            The simulation tick being evaluated.

        Returns
        -------
        :
            The result value.
        """

    @final
    def defer(
        self,
        other: "SimulationTickFunction[DeferResultT]",
        tick: Tick,
        scope: GeoScope | None = None,
        time_frame: TimeFrame | None = None,
    ) -> DeferResultT:
        """
        Defer processing to another instance of a `SimulationTickFunction`, returning
        the result of evaluation.

        This function is generic in the type of result returned by the function
        to which we are deferring (`DeferResultT`).

        Parameters
        ----------
        other :
            The other function to defer to.
        tick :
            The simulation tick being evaluated.
        scope :
            Override the geo scope for evaluation; if None, use the same scope.
        time_frame :
            Override the time frame for evaluation; if None, use the same time frame.

        Returns
        -------
        :
            The result value.
        """
        return self.defer_context(other, scope, time_frame).evaluate(tick)
