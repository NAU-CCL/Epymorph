"""Simulation parameter handling."""
import inspect
from typing import Any, Callable, Mapping, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.code import (compile_function, has_function_structure,
                           parse_function)
from epymorph.error import CompilationException
from epymorph.geo.geo import Geo
from epymorph.simulation import epymorph_namespace

ParamBase = int | float | str
ParamList = list[Union[ParamBase, 'ParamList']]
ParamPy = ParamBase | ParamList
ParamNp = NDArray[np.int64 | np.float64 | np.str_]
ParamFunction = Callable[[int, int], ParamPy | ParamNp]
ParamValue = ParamPy | ParamNp | ParamFunction
"""
Types for raw parameter values. Users can supply any of these forms when constructing
simulation parameters.
"""

Params = Mapping[str, ParamValue]
"""Simulation parameters in their non-normalized input form."""

ContextParams = dict[str, ParamNp]
"""Simulation parameters in their fully-normalized form."""


def normalize_params(raw_params: Params, geo: Geo, duration: int,
                     dtypes: Mapping[str, DTypeLike] | None = None) -> dict[str, ParamNp]:
    """
    Normalize raw parameter values to numpy arrays. dtypes can be enforced
    by passing a mapping from attribute name to the desired dtype.
    Any parameters that are already in the form of a numpy array will be copied.
    """
    def _get_dtype(name: str) -> DTypeLike | None:
        return None if dtypes is None else dtypes.get(name, None)

    global_namespace = epymorph_namespace() | {'geo': geo}

    def _norm(name: str, raw: ParamValue) -> ParamNp:
        dtype = _get_dtype(name)

        if isinstance(raw, str) and has_function_structure(raw):
            f = parse_function(raw)
            raw = compile_function(f, global_namespace)

        if callable(raw):
            return _evaluate_param_function(raw, geo.nodes, duration, dtype)
        if isinstance(raw, np.ndarray):
            return raw.astype(dtype=dtype, copy=True)
        return np.asarray(raw, dtype=dtype)

    return {k: _norm(k, v) for k, v in raw_params.items()}


def _evaluate_param_function(function: ParamFunction, nodes: int, duration: int,
                             dtype: DTypeLike | None) -> NDArray:
    """
    Evaluate a parameter function and return the result as a numpy array.
    Evaluation is based on the function signature: functions must specify
    two parameters, `t` and `n`, but may replace one or both with names
    starting with underscore to indicate that the value does not vary in that
    dimension. For example, a time-varying but node-constant function would
    define a function as `def my_param(t, _)` or `def my_param(t, _arbitrary_name)`.
    Constant values should use leading-underscore names for both parameters.
    """
    try:
        # If a param name starts with underscore, replace it with underscore.
        # This makes pattern matching is easier.
        signature = tuple(
            '_' if param.startswith('_') else param
            for param in inspect.signature(function).parameters
        )

        ignore: Any = None
        match signature:
            # Handle the different acceptable function signatures.
            case ('_', '_'):
                result = function(ignore, ignore)
            case ('t', '_'):
                result = [function(d, ignore) for d in range(duration)]
            case ('_', 'n'):
                result = [function(ignore, n) for n in range(nodes)]
            case ('t', 'n'):
                result = [[function(d, n) for n in range(nodes)]
                          for d in range(duration)]
            # Handle unsupported function signatures.
            case (_, _):
                msg = f"Unsupported parameter function signature for function: '{function.__name__}':\n"\
                    "- Parameter names can only be 't', 'n', or '_'."
                raise CompilationException(msg)
            case _:
                msg = f"Unsupported parameter function signature for function: '{function.__name__}':\n"\
                    "- Function must have two parameters, as in: `def my_param(t, n):`"
                raise CompilationException(msg)

        try:
            return np.asarray(result, dtype=dtype)
        except ValueError as e:
            msg = f"Unable to convert result of parameter function '{function.__name__}':\n- {e}"
            raise CompilationException(msg) from e

    except CompilationException as e:
        raise e
    except Exception as e:
        msg = f"An error occurred while running the parameter function '{function.__name__}':\n- {e}"
        raise CompilationException(msg) from e
