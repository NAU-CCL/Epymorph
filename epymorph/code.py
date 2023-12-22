"""
Utilities for handling user code parsed from strings.
While the primary goal is to extract runnable Python functions,
it is also important to take some security measures to mitigate
the execution of malicious code.
"""
import ast
import re
from typing import Any, Callable


def has_function_structure(s: str) -> bool:
    """
    Check if a string has the structure of a function definition.

    Args:
        s: The string to check.

    Returns:
        True if the string has the structure of a function definition, False otherwise.
    """
    pattern = r"def\s+\w+\s*\([^)]*\):"
    match = re.search(pattern, s)
    return match is not None


def parse_function(code_string: str) -> ast.FunctionDef:
    """
    Parse a function from a code string, returning the function's AST.
    It will be assumed that the string contains only a single Python function definition.
    Args:
        function_string: The function string to parse.

    Returns:
        An AST.FunctionDef node representing the function.
    """

    # Parse the code string into an AST
    tree = ast.parse(code_string, '<string>', mode='exec')
    # Assuming the code string contains only a single function definition
    f_def = tree.body[0]
    if not isinstance(f_def, ast.FunctionDef):
        raise Exception("Code does not define a valid function")
    return f_def


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
        global_namespace = {}
    local_namespace: dict[str, Any] = {}
    exec(code, global_namespace, local_namespace)
    # Now our function is defined in the local namespace, retrieve it
    # TODO: it would be nice if this was typesafe in the signature of the returned function...
    return local_namespace[function_def.name]


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
        raise Exception(f"{self.__class__.__name__} is immutable.")

    def __delattr__(self, __name: str) -> None:
        raise Exception(f"{self.__class__.__name__} is immutable.")

    def to_dict_shallow(self) -> dict[str, Any]:
        """Make a shallow copy of this Namespace as a dict."""
        # This is necessary in order to pass it to exec or eval.
        # The shallow copy allows child-namespaces to remain dot-accessible.
        return object.__getattribute__(self, '_data').copy()
