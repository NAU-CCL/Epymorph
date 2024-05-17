"""
Utilities for handling user code parsed from strings.
While the primary goal is to extract runnable Python functions,
it is also important to take some security measures to mitigate
the execution of malicious code.
"""
import ast
import re
import textwrap
from functools import partial
from typing import Any, Callable

import numpy as np
from numpy.typing import DTypeLike

from epymorph.util import pairwise_haversine, row_normalize


def has_function_structure(string: str) -> bool:
    """Check if a string seems to have the structure of a function definition."""
    return re.search(r"^\s*def\s+\w+\s*\(.*?\):", string, flags=re.MULTILINE) is not None


def parse_function(code_string: str, unsafe: bool = False) -> ast.FunctionDef:
    """
    Parse a function from a code string, returning the function's AST.
    The resulting AST will have security mitigations applied, unless `unsafe` is True.
    The string must contain a single top-level Python function definition,
    or else ValueError is raised. Raises SyntaxError if the function is not valid Python. 
    """
    tree = ast.parse(textwrap.dedent(code_string), '<string>', mode='exec')
    functions = [statement for statement in tree.body
                 if isinstance(statement, ast.FunctionDef)]
    if (n := len(functions)) != 1:
        msg = f"Code must contain exactly one top-level function definition: found {n}"
        raise ValueError(msg)
    return functions[0] if unsafe else scrub_function(functions[0])


class CodeCompileException(Exception):
    """An exception raised when code cannot be compiled for some reason."""


class CodeSecurityException(CodeCompileException):
    """An exception raised when code cannot be safely compiled due to security rules."""


_FORBIDDEN_NAMES = frozenset({
    'eval', 'exec', 'compile', 'object', 'print', 'open',
    'quit', 'exit', 'globals', 'locals', 'help', 'breakpoint'
})
"""Names which should not exist in a user-defined function."""


class SecureTransformer(ast.NodeTransformer):
    """AST transformer for applying basic security mitigations."""

    def visit_Import(self, _node: ast.Import) -> Any:
        """Silently remove imports."""
        return None

    def visit_ImportFrom(self, _node: ast.ImportFrom) -> Any:
        """Silently remove imports."""
        return None

    def visit_Name(self, node: ast.Name) -> Any:
        """No referencing sensitive names like eval or exec, or anything starting with an underscore."""
        if node.id.startswith('_'):
            raise CodeSecurityException(f"Illegal reference to `{node.id}`.")
        if node.id in _FORBIDDEN_NAMES:
            raise CodeSecurityException(f"Illegal reference to `{node.id}`.")
        return super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Disallow accessing potentially sensitive attributes (any with a leading underscore)."""
        if node.attr.startswith('_'):
            msg = f"Illegal reference to attribute `{node.attr}`."
            raise CodeSecurityException(msg)
        return super().generic_visit(node)


def scrub_function(function_def: ast.FunctionDef) -> ast.FunctionDef:
    """
    Applies security mitigations to an AST, returning the transformed AST.
    """
    return SecureTransformer().visit(function_def)


def compile_function(function_def: ast.FunctionDef, global_namespace: dict[str, Any] | None) -> Callable:
    """
    Compile the given function's AST using the given global namespace.
    Returns the function.
    Args:
        function_definition: The function definition to compile.
        global_vars: A dictionary of global variables to make available to the compiled function.

    Returns:
        A callable object representing the compiled function.
    """

    # Compile the code and execute it, providing global and local namespaces
    module = ast.Module(body=[function_def], type_ignores=[])
    code = compile(module, '<string>', mode='exec')
    if global_namespace is None:
        global_namespace = base_namespace()
    local_namespace = dict[str, Any]()
    exec(code, global_namespace, local_namespace)
    # Now our function is defined in the local namespace, retrieve it.
    function = local_namespace[function_def.name]
    if not isinstance(function, Callable):
        msg = f"`{function_def.name}` did not compile to a callable function."
        raise CodeCompileException(msg)
    return function


class ImmutableNamespace:
    """A simple dot-accessible dictionary."""

    __slots__ = ['_data']

    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any] | None = None):
        if data is None:
            data = {}
        object.__setattr__(self, '_data', data)

    def __getattribute__(self, __name: str) -> Any:
        if __name == '_data':
            __cls = self.__class__.__name__
            raise AttributeError(f"{__cls} object has no attribute '{__name}'")
        try:
            return object.__getattribute__(self, __name)
        except AttributeError:
            data = object.__getattribute__(self, '_data')
            if __name not in data:
                __cls = self.__class__.__name__
                msg = f"{__cls} object has no attribute '{__name}'"
                raise AttributeError(msg) from None
            return data[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise AttributeError(f"{self.__class__.__name__} is immutable.")

    def __delattr__(self, __name: str) -> None:
        raise AttributeError(f"{self.__class__.__name__} is immutable.")

    def to_dict_shallow(self) -> dict[str, Any]:
        """Make a shallow copy of this Namespace as a dict."""
        # This is necessary in order to pass it to exec or eval.
        # The shallow copy allows child-namespaces to remain dot-accessible.
        return object.__getattribute__(self, '_data').copy()


def base_namespace() -> dict[str, Any]:
    """Make a safer namespace for user-defined functions."""
    return {'__builtins__': {}}


def epymorph_namespace(sim_dtype: DTypeLike) -> dict[str, Any]:
    """
    Make a safe namespace for user-defined functions,
    including utilities that functions might need in epymorph.
    """
    return {
        'SimDType': sim_dtype,
        # our utility functions
        'pairwise_haversine': pairwise_haversine,
        'row_normalize': row_normalize,
        # numpy namespace
        'np': ImmutableNamespace({
            # numpy utility functions
            'array': partial(np.array, dtype=sim_dtype),
            'zeros': partial(np.zeros, dtype=sim_dtype),
            'zeros_like': partial(np.zeros_like, dtype=sim_dtype),
            'ones': partial(np.ones, dtype=sim_dtype),
            'ones_like': partial(np.ones_like, dtype=sim_dtype),
            'full': partial(np.full, dtype=sim_dtype),
            'arange': partial(np.arange, dtype=sim_dtype),
            'concatenate': partial(np.concatenate, dtype=sim_dtype),
            'sum': partial(np.sum, dtype=sim_dtype),
            'newaxis': np.newaxis,
            'fill_diagonal': np.fill_diagonal,
            # numpy math functions
            'radians': np.radians,
            'degrees': np.degrees,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'arctan2': np.arctan2,
            'sqrt': np.sqrt,
            'add': np.add,
            'subtract': np.subtract,
            'multiply': np.multiply,
            'divide': np.divide,
            'maximum': np.maximum,
            'minimum': np.minimum,
            'absolute': np.absolute,
            'floor': np.floor,
            'ceil': np.ceil,
            'pi': np.pi,
        }),
        **base_namespace(),
    }
