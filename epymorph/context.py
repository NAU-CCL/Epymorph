"""
The need for certain info about a simulation cuts across modules (ipm, movement, geo), so
the SimContext structure is here to contain that info and avoid circular dependencies.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.clock import Clock
from epymorph.data_shape import SimDimension
from epymorph.geo.abstract import _ProxyGeo, proxy
from epymorph.geo.geo import Geo
from epymorph.util import (compile_function, has_function_structure,
                           make_namespace, parse_function,
                           preprocess_signature)

SimDType = np.int64
"""
This is the numpy datatype that should be used to represent internal simulation data.
Where segments of the application maintain compartment and/or event counts,
they should take pains to use this type at all times (if possible).

WARNING: before adjusting this see the note in `CompartmentModelIpm._rate_args(...)`.
"""

# SimDType being centrally-located means we can change it reliably.

Compartments = NDArray[SimDType]
"""Alias for ndarrays representing compartment counts."""

Events = NDArray[SimDType]
"""Alias for ndarrays representing event counts."""

# Aliases (hopefully) make it a bit easier to keep all these NDArrays sorted out.


# DataDict


def normalize_lists(data: dict[str, Any], dtypes: dict[str, DTypeLike] | None = None) -> dict[str, Any]:
    """
    Normalize a dictionary of values so that all lists are replaced with numpy arrays.
    If you would like to force certain values to take certain dtypes, provide the `dtypes` argument 
    with a mapping from key to dtype (types will not affect non-list values).
    """
    if dtypes is None:
        dtypes = {}
    ps = dict[str, Any]()
    # Replace list values with numpy arrays.
    for key, value in data.items():
        if isinstance(value, list):
            dt = dtypes.get(key, None)
            ps[key] = np.asarray(value, dtype=dt)
        else:
            ps[key] = value
    return ps


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

    parameter_arrays = dict[str, NDArray]()
    compartments = geo.nodes
    p = cast(_ProxyGeo, proxy)
    p.set_actual_geo(geo)

    global_namespace = make_namespace(geo=geo, SimDType=SimDType)

    for key, value in data.items():

        dt = dtypes.get(key, None)

        if callable(value):
            parameter_arrays[key] = evaluate_function(
                value, compartments, duration, dt)
        elif isinstance(value, str) and has_function_structure(str(value)):
            function_definition = parse_function(value)
            compiled_function = compile_function(function_definition, global_namespace)
            parameter_arrays[key] = evaluate_function(
                compiled_function, compartments, duration, dt)
        else:
            parameter_arrays[key] = np.asarray(value, dtype=dtypes.get(key, None))

    return parameter_arrays


def evaluate_function(function: callable, compartments: int, duration: int, dt: DTypeLike | None = None) -> NDArray:  # type:ignore
    """
    Evaluate a function and return the result as a numpy array.

    Args:
        function: The function to evaluate.
        compartments: The number of compartments in the system.
        duration: The duration of the simulation.
        dt: The data type for the result of the function evaluation.

    Returns:
        A numpy array representing the result of the evaluation.
    """

    signature = tuple(inspect.signature(function).parameters.keys())
    processed_signature = preprocess_signature(signature)
    try:
        # Handle different cases based on the function signature
        if processed_signature == ('_', '_'):
            result = function(None, None)
        elif processed_signature == ('t', '_'):
            result = [function(d, None) for d in range(duration)]
        elif processed_signature == ('_', 'n'):
            result = [function(None, c) for c in range(compartments)]
        elif processed_signature == ('t', 'n'):
            result = [[function(d, c) for c in range(compartments)]
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


# SimContext


@dataclass(frozen=True)
class SimContext(SimDimension):
    """Metadata about the simulation being run."""

    geo: Geo
    # ipm info
    compartments: int
    compartment_tags: list[list[str]]
    events: int
    # run info
    param: dict[str, NDArray]
    clock: Clock
    rng: np.random.Generator
    # denormalized info
    nodes: int = field(init=False)
    ticks: int = field(init=False)
    days: int = field(init=False)
    TNCE: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'nodes', self.geo.nodes)
        object.__setattr__(self, 'ticks', self.clock.num_ticks)
        object.__setattr__(self, 'days', self.clock.num_days)
        tnce = (self.clock.num_ticks, self.nodes,
                self.compartments, self.events)
        object.__setattr__(self, 'TNCE', tnce)
