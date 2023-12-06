"""
A fully-configured RUME has a context that can be used to interact
with simulation data, for example, accessing geo and parameter attributes,
calculating the simulation clock, initializing the world state, and so on.
"""
import inspect
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import partial
from typing import Callable, Iterable, NamedTuple, Self, cast

import numpy as np
from numpy.typing import NDArray

from epymorph.attribute import AttributeDef, AttributeTypeNp
from epymorph.compartment_model import CompartmentModel
from epymorph.error import (AttributeException, InitException,
                            IpmValidationException)
from epymorph.geo.abstract import proxy_geo
from epymorph.geo.geo import Geo
from epymorph.initializer import (InitContext, Initializer,
                                  normalize_init_params)
from epymorph.movement.movement_model import MovementModel
from epymorph.movement.parser import MovementSpec
from epymorph.params import ContextParams, ParamNp, Params, normalize_params
from epymorph.simulation import (SimDimensions, SimDType, Tick, TickDelta,
                                 TimeFrame)
from epymorph.util import MemoDict, NumpyTypeError, check_ndarray


class RumeConfig(NamedTuple):
    """The set of info needed to build a RUME."""
    geo: Geo
    ipm: CompartmentModel
    mm: MovementModel | MovementSpec
    params: Params
    time_frame: TimeFrame
    initializer: Initializer
    rng: Callable[[], np.random.Generator]


AttributeValue = AttributeTypeNp | NDArray[AttributeTypeNp]
AttributeGetter = Callable[[Tick, int], AttributeValue]


@dataclass
class RumeContext:
    """The fully-realized configuration and data we need to run a simulation."""
    dim: SimDimensions
    geo: Geo
    ipm: CompartmentModel
    mm: MovementModel | MovementSpec  # NOTE: temporary
    params: ContextParams
    raw_params: Params  # need params in their original form for initializer
    time_frame: TimeFrame
    initializer: Initializer
    rng: np.random.Generator

    _attribute_getters: MemoDict[AttributeDef, AttributeGetter] = field(init=False)

    @classmethod
    def from_config(cls, config: RumeConfig) -> Self:
        """Construct a RumeContext from a RumeConfig."""
        match config.mm:
            case MovementModel():
                tau_step_lengths = config.mm.tau_steps
            case MovementSpec():
                tau_step_lengths = config.mm.steps.step_lengths

        dim = SimDimensions.build(
            tau_step_lengths=tau_step_lengths,
            days=config.time_frame.duration_days,
            nodes=config.geo.nodes,
            compartments=config.ipm.num_compartments,
            events=config.ipm.num_events
        )

        # Collect expected attribute types for normalization.
        # NOTE: this will require much more thought when the MM can declare its required params.
        # How should we reconcile conflicting type declarations?
        attr_dtypes = {
            a.name: a.dtype_as_np
            for a in config.ipm.attributes
        }

        with proxy_geo(config.geo):
            # Parameters might be functions reference the proxy geo,
            # so to evaluate them we must be in the `proxy_geo` context.
            ctx_params = normalize_params(
                config.params,
                config.geo,
                config.time_frame.duration_days,
                attr_dtypes
            )

        return cls(dim, config.geo, config.ipm, config.mm,
                   ctx_params, config.params, config.time_frame,
                   config.initializer, config.rng())

    def __post_init__(self):
        def _create_attribute_getter(attr: AttributeDef) -> AttributeGetter:
            """Create a tick-and-node accessor function for the given attribute."""
            data_raw = self._get_attribute_value(attr)
            data = attr.shape.adapt(self.dim, data_raw, True)
            if data is None:
                msg = f"Attribute '{attr.name}' could not be adpated to the required shape."
                raise AttributeException(msg)
            return attr.shape.accessor(data)

        self._attribute_getters = MemoDict(_create_attribute_getter)

    def clock(self) -> Iterable[Tick]:
        """Generate the simulation clock signal: a series of Tick objects describing each time step."""
        return _simulation_clock(self.dim, self.time_frame.start_date)

    def resolve_tick(self, tick: Tick, delta: TickDelta) -> int:
        """Add a delta to a tick to get the index of the resulting tick."""
        return -1 if delta.days == -1 else \
            tick.index - tick.step + (self.dim.tau_steps * delta.days) + delta.step

    def update_param(self, name: str, value: ParamNp) -> None:
        """Updates a params value."""
        self.params[name] = value.copy()
        attrs = [a for a in self._attribute_getters if a.name == name]
        for a in attrs:
            del self._attribute_getters[a]

    def _get_attribute_value(self, attr: AttributeDef) -> NDArray:
        """Retrieve the value associated with the given attribute."""
        name = attr.name
        match attr.source:
            case 'geo':
                if not name in self.geo.spec.attribute_map:
                    msg = f"Missing geo attribute '{name}'"
                    raise AttributeException(msg)
                return self.geo[name]
            case 'params':
                if not name in self.params:
                    msg = f"Missing params attribute '{name}'"
                    raise AttributeException(msg)
                return self.params[name]

    def get_attribute(self, attr: AttributeDef, tick: Tick, node: int) -> AttributeValue:
        """Get an attribute value at a specific tick and node."""
        return self._attribute_getters[attr](tick, node)

    def validate_ipm(self) -> None:
        """Validate that this context provides the attributes required by the RUME IPM."""
        # Collect all attribute errors to raise as a group.
        errors = []
        for attr in self.ipm.attributes:
            try:
                value = self._get_attribute_value(attr)

                if not attr.shape.matches(self.dim, value, True):
                    msg = f"Attribute '{attr.name}' was expected to be an array of shape {attr.shape} " + \
                        f"-- got {value.shape}."
                    raise AttributeException(msg)

                if not attr.dtype_as_np.type == np.dtype(value.dtype).type:
                    msg = f"Attribute '{attr.name}' was expected to be an array of type {attr.dtype} " +\
                        f"-- got {value.dtype}."
                    raise AttributeException(msg)
            except AttributeException as e:
                errors.append(e)

        if len(errors) > 0:
            msg = "IPM attribute requirements were not met. See errors:" + \
                "".join(f"\n- {e}" for e in errors)
            raise IpmValidationException(msg)

    def initialize(self) -> NDArray[SimDType]:
        """Executes an initializer, attempting to fill arguments from the context."""
        # NOTE: initializers and partials generated from them should exclusively use keyword arguments
        return _initialize(self)


def _simulation_clock(dim: SimDimensions, start_date: date) -> Iterable[Tick]:
    """Generator for the sequence of ticks which makes up the simulation clock."""
    one_day = timedelta(days=1)
    tau_steps = list(enumerate(dim.tau_step_lengths))
    curr_index = 0
    curr_date = start_date
    for day in range(dim.days):
        for step, tau in tau_steps:
            yield Tick(curr_index, day, curr_date, step, tau)
            curr_index += 1
        curr_date += one_day


def _initialize(ctx: RumeContext) -> NDArray[SimDType]:
    """
    Executes the initialization logic for a RumeContext.
    Much of the processing here is an attempt to auto-wire initializer
    function parameters from the context -- providing either the context itself,
    a geo attribute, a params attribute, or a default argument -- if they have
    not otherwise been provided using partial.
    """
    init = ctx.initializer
    partial_kwargs = set[str]()
    if isinstance(init, partial):
        # partial funcs require a bit of extra massaging
        init_name = init.func.__name__
        partial_kwargs = set(init.keywords)
        init = cast(Initializer, init)
    else:
        init_name = cast(str, getattr(init, '__name__', 'UNKNOWN'))

    init_params = normalize_init_params({**ctx.raw_params})

    # get list of args for function
    sig = inspect.signature(init)

    # Build up the arguments dict.
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in ['POSITIONAL_ONLY', 'VAR_POSITIONAL', 'VAR_KEYWORD']:
            # var-args not supported
            msg = f"'{init_name}' requires an argument of an unsupported kind: {p.name} is a {p.kind} parameter"
            raise InitException(msg)

        if p.name in partial_kwargs:
            # Skip any args in partial_kwargs, otherwise we're just repeating them.
            continue
        elif p.name == 'ctx' or p.annotation in [RumeContext, InitContext]:
            # If context needed, supply context!
            kwargs[p.name] = ctx
        elif p.name in ctx.geo.spec.attribute_map:
            # If name is in geo, use that.
            kwargs[p.name] = ctx.geo[p.name]
        elif p.name in init_params:
            # If name is in params, use that.
            kwargs[p.name] = init_params[p.name]
        elif p.default is not inspect.Parameter.empty:
            # If arg has a default, use that.
            kwargs[p.name] = p.default
        else:
            # Unable to auto-wire the arg!
            msg = f"'{init_name}' requires an argument that we couldn't auto-wire: {p.name} ({p.annotation})"
            raise InitException(msg)

    # Execute function with matched args.
    try:
        result = init(**kwargs)
    except InitException as e:
        raise e
    except Exception as e:
        raise InitException('Initializer failed during execution.') from e

    if np.min(result) < 0:
        raise InitException(f"Initializer '{init_name}' returned values less than zero")

    try:
        _, N, C, _ = ctx.dim.TNCE
        check_ndarray(result, SimDType, (N, C))
    except NumpyTypeError as e:
        raise InitException(f"Invalid return type from '{init_name}'") from e
    return result
