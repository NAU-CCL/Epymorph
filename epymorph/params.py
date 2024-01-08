"""Simulation parameter handling."""
import ast
import inspect
from typing import Any, Callable, Mapping, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from epymorph.code import (compile_function, has_function_structure,
                           parse_function)
from epymorph.error import CompilationException
from epymorph.geo.geo import Geo
from epymorph.simulation import SimDimensions, epymorph_namespace

ParamBase = int | float | str
ParamList = list[Union[ParamBase, 'ParamList']]  # (recursively-nested lists)
ParamPy = ParamBase | ParamList

ParamNpBase = np.int64 | np.float64 | np.str_
ParamNp = NDArray[ParamNpBase]

ParamFunction = Callable[[int, int], ParamBase | ParamNpBase]

ParamValue = ParamPy | ParamNp | ParamFunction
"""
Types for raw parameter values. Users can supply any of these forms when constructing
simulation parameters.
"""

Params = Mapping[str, ParamValue]
"""Simulation parameters in their non-normalized input form."""

ContextParams = dict[str, ParamNp]
"""Simulation parameters in their fully-normalized form."""


def normalize_params(raw_params: Params, geo: Geo, dim: SimDimensions,
                     dtypes: Mapping[str, DTypeLike] | None = None) -> dict[str, ParamNp]:
    """
    Normalize raw parameter values to numpy arrays. dtypes can be enforced
    by passing a mapping from attribute name to the desired dtype.
    Any parameters that are already in the form of a numpy array will be copied.
    """
    def _get_dtype(name: str) -> DTypeLike | None:
        return None if dtypes is None else dtypes.get(name, None)

    def _norm(name: str, raw: ParamValue) -> ParamNp:
        dtype = _get_dtype(name)

        if isinstance(raw, str) and has_function_structure(raw):
            f = compile_str_function(raw)
            return _evaluate_param_function(f, geo, dim, dtype)
        if callable(raw):
            f = compile_py_function(raw)
            return _evaluate_param_function(f, geo, dim, dtype)
        if isinstance(raw, np.ndarray):
            return raw.astype(dtype=dtype, copy=True)
        return np.asarray(raw, dtype=dtype)

    return {k: _norm(k, v) for k, v in raw_params.items()}


class ParamFunctionContext:
    """The subset of the RumeContext that parameter functions have access to."""
    geo: Geo
    dim: SimDimensions

    def __init__(self, geo: Geo, dim: SimDimensions):
        self.geo = geo
        self.dim = dim


CompiledParamFunction = Callable[[
    ParamFunctionContext, int, int], ParamBase | ParamNpBase]


def _evaluate_param_function(function: CompiledParamFunction, geo: Geo, dim: SimDimensions,
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
        ctx = ParamFunctionContext(geo, dim)

        # If a param name starts with underscore, replace it with underscore.
        # This makes pattern matching is easier.
        signature = tuple(
            '_' if param.startswith('_') else param
            for param in inspect.signature(function).parameters
        )

        ignore: Any = None
        match signature:
            # Handle the different acceptable function signatures.
            case ('ctx', '_', '_'):
                result = function(ctx, ignore, ignore)
            case ('ctx', 't', '_'):
                result = [function(ctx, d, ignore) for d in range(dim.days)]
            case ('ctx', '_', 'n'):
                result = [function(ctx, ignore, n) for n in range(dim.nodes)]
            case ('ctx', 't', 'n'):
                result = [[function(ctx, d, n) for n in range(dim.nodes)]
                          for d in range(dim.days)]
            # Handle unsupported function signatures.
            case (_, _, _):
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


class ParamFunctionTransformer(ast.NodeTransformer):
    """
    Transforms parameter function code so that we can pass geo
    via function arguments instead of the namespace. The goal is to
    simplify the function interface for end users.

    A function like:

    def my_param(t, n):
        return geo['population'][n] / dim.nodes

    Will be rewritten as:

    def my_param(ctx, t, n):
        return ctx.geo['population'][n] / ctx.dim.nodes
    """

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Modify references to dictionaries that should be in context."""
        if isinstance(node.value, ast.Name):
            if node.value.id in ['geo']:
                node.value = ast.Attribute(
                    value=ast.Name(id='ctx', ctx=ast.Load()),
                    attr=node.value.id,
                    ctx=ast.Load(),
                )
                return node
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Modify references to objects that should be in context."""
        if isinstance(node.value, ast.Name):
            if node.value.id in ['dim']:
                node.value = ast.Attribute(
                    value=ast.Name(id='ctx', ctx=ast.Load()),
                    attr=node.value.id,
                    ctx=ast.Load(),
                )
                return node
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Modify function parameters."""
        new_node = self.generic_visit(node)
        if isinstance(new_node, ast.FunctionDef):
            geo_arg = ast.arg(
                arg='ctx',
                annotation=ast.Name(id='ParamFunctionContext', ctx=ast.Load()),
            )
            args = [geo_arg, *new_node.args.args]
            new_node.args.args = args
        return new_node


def transform_param_ast(orig_ast: ast.FunctionDef) -> ast.FunctionDef:
    """Transforms movement clause code. See MovementFunctionTransformer for details."""
    trns_ast = ParamFunctionTransformer().visit(orig_ast)
    ast.fix_missing_locations(trns_ast)
    return trns_ast


def _param_global_namespace() -> dict[str, Any]:
    """Construct the namespace for compiling a parameter function."""
    return epymorph_namespace() | {'ParamFunctionContext': ParamFunctionContext}


def compile_str_function(param_code: str) -> CompiledParamFunction:
    """
    Compile a param given as the text of a Python function.
    We will perform AST manipulations on the function before returning a Callable.
    """
    func_ast = parse_function(param_code)
    trns_ast = transform_param_ast(func_ast)
    return compile_function(trns_ast, _param_global_namespace())


def compile_py_function(param_func: Callable) -> CompiledParamFunction:
    """
    Re-compile a param given as a Python Callable.
    This allows us to do our AST manipulations.
    """
    param_code = inspect.getsource(param_func)
    func_ast = parse_function(param_code)
    trns_ast = transform_param_ast(func_ast)
    # Modify the function's original globals.
    # This way, any (non-conflicting) variables passed via closure will still be available.
    global_namespace = param_func.__globals__ | _param_global_namespace()
    return compile_function(trns_ast, global_namespace)
