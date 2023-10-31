import inspect
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Callable, Iterable, NamedTuple, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

import epymorph.data_shape as ds
from epymorph.attribute import AttributeDef, AttributeTypeNp
from epymorph.compartment_model import CompartmentModel
from epymorph.error import (AttributeException, InitException,
                            IpmValidationException)
from epymorph.geo.abstract import _ProxyGeo, proxy
from epymorph.geo.geo import Geo
from epymorph.initializer import Initializer
from epymorph.movement_model import MovementModel, compile_spec
from epymorph.parser.movement import MovementSpec
from epymorph.simulation import (Params, SimDimensions, SimDType, Tick,
                                 TickDelta, TimeFrame, base_namespace)
from epymorph.util import (MemoDict, NumpyTypeError, check_ndarray,
                           compile_function, has_function_structure,
                           parse_function)

AttributeValue = AttributeTypeNp | NDArray[AttributeTypeNp]
AttributeGetter = Callable[[Tick, int], AttributeValue]


class ExecutionContext:
    """The fully-realized configuration and data we need to run a simulation."""
    dim: SimDimensions
    geo: Geo
    ipm: CompartmentModel
    mm: MovementModel
    params: Params
    time_frame: TimeFrame
    initializer: Initializer
    rng: np.random.Generator

    _attribute_getters: MemoDict[AttributeDef, AttributeGetter]

    def __init__(self,
                 dim: SimDimensions,
                 geo: Geo,
                 ipm: CompartmentModel,
                 mm: MovementModel,
                 params: Params,
                 time_frame: TimeFrame,
                 initials: Initializer,
                 rng: np.random.Generator):
        self.dim = dim
        self.geo = geo
        self.ipm = ipm
        self.mm = mm
        self.params = params
        self.time_frame = time_frame
        self.initializer = initials
        self.rng = rng
        self._attribute_getters = MemoDict(self._create_attribute_getter)

    def clock(self) -> Iterable[Tick]:
        """Generate the simulation clock signal: a series of Tick objects describing each time step."""
        one_day = timedelta(days=1)
        days = self.dim.days
        tau_steps = self.dim.tau_steps
        curr_date = self.time_frame.start_date
        for index in range(days * tau_steps):
            day, step = divmod(index, tau_steps)
            tau = self.dim.tau_step_lengths[step]
            yield Tick(index, day, curr_date, step, tau)
            curr_date += one_day

    def resolve_tick(self, tick: Tick, delta: TickDelta) -> int:
        """Add a delta to a tick to get the index of the resulting tick."""
        return -1 if delta.days == -1 else \
            tick.index - tick.step + (self.dim.tau_steps * delta.days) + delta.step

    def update_params(self, params: Params) -> None:
        # TODO: is this necessary? test if mutation of the underlying data is possible
        # without having to re-create the associated getter...
        # might be fine as long as the data array can changed in-place
        self.params = params
        self._attribute_getters.clear()

    def _get_attribute_value(self, attr: AttributeDef) -> NDArray:
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

    def _create_attribute_getter(self, attr: AttributeDef) -> AttributeGetter:
        data_raw = self._get_attribute_value(attr)
        data = attr.shape.adapt(self.dim, data_raw, True)
        if data is None:
            msg = f"Attribute '{attr.name}' could not be adpated to the required shape."
            raise AttributeException(msg)

        match attr.shape:
            case ds.Scalar():
                return lambda tick, node: data  # type: ignore
            case ds.Time():
                return lambda tick, node: data[tick.day]
            case ds.Node():
                return lambda tick, node: data[node]
            case ds.TimeAndNode():
                return lambda tick, node: data[tick.day, node]
            case ds.Arbitrary(a):
                return lambda tick, node: data[a]
            case ds.TimeAndArbitrary(a):
                return lambda tick, node: data[tick.day, a]
            case ds.NodeAndArbitrary(a):
                return lambda tick, node: data[node, a]
            case ds.TimeAndNodeAndArbitrary(a):
                return lambda tick, node: data[tick.day, node, a]
            case _:
                msg = f"Unsupported shape: {attr.shape}"
                raise AttributeException(msg)

    def validate_ipm(self) -> None:
        errors = []
        for ipm_attr in self.ipm.attributes:
            try:
                attr = ipm_attr.attribute
                value = self._get_attribute_value(attr)

                if not attr.shape.matches(self.dim, value, ipm_attr.allow_broadcast):
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

    def get_attribute(self, attr: AttributeDef, tick: Tick, node: int) -> AttributeValue:
        return self._attribute_getters[attr](tick, node)

    def initialize(self) -> NDArray[SimDType]:
        """Executes an initializer, attempting to fill arguments from the context."""
        # NOTE: initializers and partials generated from them should exclusively use keyword arguments

        ctx = self
        init = self.initializer
        partial_kwargs = set[str]()
        if isinstance(init, partial):
            # partial funcs require a bit of extra massaging
            init_name = init.func.__name__
            partial_kwargs = set(init.keywords.keys())
            init = cast(Initializer, init)
        else:
            init_name = cast(str, getattr(init, '__name__', 'UNKNOWN'))

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
            elif p.name == 'ctx' or p.annotation == ExecutionContext:
                # If context needed, supply context!
                kwargs[p.name] = ctx
            elif p.name in ctx.geo.spec.attribute_map:
                # If name is in geo, use that.
                kwargs[p.name] = ctx.geo[p.name]
            elif p.name in ctx.params:
                # If name is in params, use that.
                kwargs[p.name] = ctx.params[p.name]
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

        try:
            _, N, C, _ = ctx.dim.TNCE
            check_ndarray(result, SimDType, (N, C))
        except NumpyTypeError as e:
            raise InitException(f"Invalid return type from '{init_name}'") from e
        return result


class ExecutionConfig(NamedTuple):
    """The set of info needed to build an execution context."""
    geo: Geo
    ipm: CompartmentModel
    mm: MovementModel | MovementSpec
    params: Params
    time_frame: TimeFrame
    initializer: Initializer
    rng: Callable[[], np.random.Generator]


def build_execution_context(config: ExecutionConfig) -> ExecutionContext:
    """Construct an ExecutionContext from an ExecutionConfig."""
    geo, ipm, mm, params, time_frame, initializer, rng = config
    match mm:
        case MovementModel():
            tau_step_lengths = mm.tau_steps
        case MovementSpec():
            tau_step_lengths = mm.steps.step_lengths

    dim = SimDimensions.build(
        tau_step_lengths=tau_step_lengths,
        days=time_frame.duration_days,
        nodes=geo.nodes,
        compartments=ipm.num_compartments,
        events=ipm.num_events
    )

    actual_rng = rng()

    # If we were given a MovementSpec, we need to compile it now.
    if isinstance(mm, MovementSpec):
        @dataclass
        class _MmCtx:
            dim: SimDimensions
            geo: Geo
            ipm: CompartmentModel
            params: Params
            rng: np.random.Generator

        mm_ctx = _MmCtx(dim, geo, ipm, params, actual_rng)
        mm = compile_spec(mm_ctx, mm)

    return ExecutionContext(dim, geo, ipm, mm, params, time_frame, initializer, actual_rng)


def normalize_params(data: dict[str, Any], geo: Geo, duration: int, dtypes: dict[str, DTypeLike] | None = None) -> dict[str, NDArray]:
    """
    Normalize a dictionary of values so that all lists are replaced with numpy arrays.

    Args:
        data: A dictionary of values to normalize.
        compartments: The number of compartments in the system.
        duration: The duration of the simulation.
        dtypes: A dictionary of data types for the parameters.

    Returns:
        A dictionary of numpy arrays representing the normalized parameters.
    """
    if dtypes is None:
        dtypes = {}

    p = cast(_ProxyGeo, proxy)
    p.set_actual_geo(geo)

    global_namespace = base_namespace() | {
        'geo': geo,
    }

    parameter_arrays = dict[str, NDArray]()
    for key, value in data.items():
        dt = dtypes.get(key, None)

        if callable(value):
            parameter_arrays[key] = evaluate_function(
                value, geo.nodes, duration, dt)
        elif isinstance(value, str) and has_function_structure(value):
            function_definition = parse_function(value)
            compiled_function = compile_function(function_definition, global_namespace)
            parameter_arrays[key] = evaluate_function(
                compiled_function, geo.nodes, duration, dt)
        else:
            parameter_arrays[key] = np.asarray(value, dtype=dtypes.get(key, None))

    return parameter_arrays


def evaluate_function(function: Callable, nodes: int, duration: int, dt: DTypeLike | None = None) -> NDArray:
    """
    Evaluate a function and return the result as a numpy array.

    Args:
        function: The function to evaluate.
        nodes: The number of nodes in the system.
        duration: The duration of the simulation.
        dt: The data type for the result of the function evaluation.

    Returns:
        A numpy array representing the result of the evaluation.
    """

    signature = tuple(inspect.signature(function).parameters.keys())
    processed_signature = tuple('_' if param.startswith('_')
                                else param for param in signature)
    try:
        # Handle different cases based on the function signature
        if processed_signature == ('_', '_'):
            result = function(None, None)
        elif processed_signature == ('t', '_'):
            result = [function(d, None) for d in range(duration)]
        elif processed_signature == ('_', 'n'):
            result = [function(None, n) for n in range(nodes)]
        elif processed_signature == ('t', 'n'):
            result = [[function(d, n) for n in range(nodes)]
                      for d in range(duration)]
        else:
            # Handle unsupported function signatures
            if len(signature) != 2:
                raise ValueError(
                    f"Unsupported function signature for function: {function.__name__}. Function must have two parameters, def func_name(_, _)")
            else:
                raise ValueError(
                    f"Unsupported function signature for function: {function.__name__}. Parameter names can only be 't', 'n', or '_'.")

    except (IndexError, ValueError, IndentationError) as e:
        raise ValueError(
            f"An error occurred while running the parameter function '{function.__name__}': {str(e)}") from e

    return np.asarray(result, dtype=dt)
