import ast
import re
from typing import Any, Callable, Generic, Iterable, TypeVar

import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

# epymorph common types


Compartments = NDArray[np.int_]
Events = NDArray[np.int_]
DataDict = dict[str, Any]


# function utilities


T = TypeVar('T')


def identity(x: T) -> T:
    return x


def constant(x: T) -> Callable[..., T]:
    return lambda *_: x


# numpy utilities


N = TypeVar('N', bound=np.number)

NumpyIndices = NDArray[np.int_]


def stutter(it: Iterable[T], times: int) -> Iterable[T]:
    """Make the iterable `it` repeat each item `times` times.
       (Unlike `itertools.repeat` which repeats whole sequences in order.)"""
    return (xs for x in it for xs in (x,) * times)


def stridesum(arr: NDArray[N], n: int) -> NDArray[N]:
    """Compute a new array by grouping every `n` rows and summing them together."""
    if len(arr) % n != 0:
        pad = n - (len(arr) % n)
        arr = np.pad(arr,
                     pad_width=(0, pad),
                     mode='constant',
                     constant_values=0)
    return arr.reshape((-1, n)).sum(axis=1)


def normalize(arr: NDArray[N]) -> NDArray[N]:
    """
    Normalize the values in an array by subtracting the min and dividing by the range.
    """
    min = arr.min()
    max = arr.max()
    return (arr - min) / (max - min)


def top(size: int, arr: NDArray) -> NumpyIndices:
    """
    Find the top `size` elements in `arr` and return their indices.
    Assumes the array is flat and the kind of thing that can be order-compared.
    """
    return np.argpartition(arr, -size)[-size:]


def bottom(size: int, arr: NDArray) -> NumpyIndices:
    """
    Find the bottom `size` elements in `arr` and return their indices.
    Assumes the array is flat and the kind of thing that can be order-compared.
    """
    return np.argpartition(arr, size)[:size]


def is_square(arr: NDArray) -> bool:
    """Is this numpy array 2 dimensions and square in shape?"""
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1]


def expand_data(data: float | int | list | NDArray, rows: int, cols: int) -> NDArray:
    """
    Take the given data and try to expand it to fit a (rows, cols) numpy array.
    If the data is already in the right shape, it is returned as-is.
    If given a scalar: all cells will contain the same value.
    If given a 1-dimensional array of size (cols): each row will contain the same values.
    If given a 1-dimensional array of size (rows): all columns will contain the same values.
    (If rows and cols are equal, you'll get repeated rows.)
    """
    if isinstance(data, list):
        data = np.array(data)
    desired_shape = (rows, cols)
    if isinstance(data, np.ndarray):
        if data.shape == desired_shape:
            return data
        elif data.shape == (cols,):
            return np.full(desired_shape, data)
        elif data.shape == (rows,):
            return np.full((cols, rows), data).T
        else:
            print(data.shape)
            raise Exception("Invalid beta parameter.A")
    elif isinstance(data, int):
        return np.full(desired_shape, data)
    elif isinstance(data, float):
        return np.full(desired_shape, data)
    else:
        raise Exception("Invalid beta parameter.B")


_duration_regex = re.compile(r"^([0-9]+)([dwmy])$", re.IGNORECASE)


def parse_duration(s: str) -> relativedelta | None:
    """Parses a duration expression like "30d" to mean 30 days. Supports days (d), weeks (w), months (m), and years (y)."""

    match = _duration_regex.search(s)
    if not match:
        return None
    else:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "d":
            return relativedelta(days=value)
        elif unit == "w":
            return relativedelta(weeks=value)
        elif unit == "m":
            return relativedelta(months=value)
        elif unit == "y":
            return relativedelta(years=value)
        else:
            return None


# console decorations


def progress(percent: float) -> str:
    """Creates a progress bar string."""
    p = 100 * max(0.0, min(percent, 1.0))
    n = int(p // 5)
    bar = ('#' * n) + (' ' * (20 - n))
    return f"|{bar}| {p:.0f}% "


# pub-sub events


class Event(Generic[T]):
    subscribers: list[Callable[[T], None]]

    def __init__(self):
        self.subscribers = []

    def subscribe(self, sub: Callable[[T], None]) -> None:
        self.subscribers.append(sub)

    def publish(self, event: T) -> None:
        for subscriber in self.subscribers:
            subscriber(event)


# AST function utilities


def parse_function(code_string: str) -> ast.FunctionDef:
    """
    Parse a function from a code string, returning the function's AST.
    It will be assumed that the string contains only a single Python function definition.
    """

    # Parse the code string into an AST
    tree = ast.parse(code_string, '<string>', mode='exec')
    # Assuming the code string contains only a single function definition
    f_def = tree.body[0]
    if not isinstance(f_def, ast.FunctionDef):
        raise Exception(f"Code does not define a valid function")
    return f_def


def compile_function(function_def: ast.FunctionDef, global_namespace: dict[str, Any] | None) -> Callable:
    """
    Compile the given function's AST using the given global namespace.
    Returns the function.
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
