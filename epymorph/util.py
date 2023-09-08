from __future__ import annotations

import ast
import re
from typing import (Any, Callable, Generic, Iterable, Literal, OrderedDict,
                    TypeGuard, TypeVar)

import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import DTypeLike, NDArray
from pydantic import BaseModel, model_serializer, model_validator

# function utilities


T = TypeVar('T')


def identity(x: T) -> T:
    return x


def constant(x: T) -> Callable[..., T]:
    return lambda *_: x


# collection utilities


def index_where(it: Iterable[T], predicate: Callable[[T], bool]) -> int:
    for i, x in enumerate(it):
        if predicate(x):
            return i
    return -1


def iterator_length(it: Iterable[Any]) -> int:
    """
    Count the number of items in the given iterator.
    Warning: this consumes the iterator! It also never terminates if the iterator is infinite.
    """
    length = 0
    for _ in it:
        length += 1
    return length


def list_not_none(it: Iterable[T]) -> list[T]:
    return [x for x in it if x is not None]


class NotUniqueException(Exception):
    def __init__(self):
        super().__init__("Collection contains non-unique values.")


def filter_unique(xs: Iterable[T]) -> list[T]:
    xset = set[T]()
    ys = list[T]()
    for x in xs:
        if x not in xset:
            ys.append(x)
            xset.add(x)
    return ys


def as_unique_set(xs: list[T]) -> set[T]:
    """Transform list to set, raising a NotUniqueException if the list contains non-unique values."""
    xs_set = set(xs)
    if len(xs_set) != len(xs):
        raise NotUniqueException()
    return xs_set


def as_list(x: T | list[T]) -> list[T]:
    """If `x` is a list, return it unchanged. If it's a single value, wrap it in a list."""
    return x if isinstance(x, list) else [x]


K, V = TypeVar('K'), TypeVar('V')


def as_sorted_dict(x: dict[K, V]) -> OrderedDict[K, V]:
    """Returns a sorted OrderedDict of the given dict."""
    return OrderedDict(sorted(x.items()))


# numpy utilities


N = TypeVar('N', bound=np.number)

NDIndices = NDArray[np.intp]


def stutter(it: Iterable[T], times: int) -> Iterable[T]:
    """Make the iterable `it` repeat each item `times` times.
       (Unlike `itertools.repeat` which repeats whole sequences in order.)"""
    return (xs for x in it for xs in (x,) * times)


def stridesum(arr: NDArray[N], n: int, dtype: DTypeLike = None) -> NDArray[N]:
    """Compute a new array by grouping every `n` rows and summing them together."""
    if len(arr) % n != 0:
        pad = n - (len(arr) % n)
        arr = np.pad(arr,
                     pad_width=(0, pad),
                     mode='constant',
                     constant_values=0)
    return arr.reshape((-1, n)).sum(axis=1, dtype=dtype)


def normalize(arr: NDArray[N]) -> NDArray[N]:
    """
    Normalize the values in an array by subtracting the min and dividing by the range.
    """
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)


def row_normalize(arr: NDArray[N], row_sums: NDArray[N] | None = None, dtype: DTypeLike = None) -> NDArray[N]:
    """
    Assuming `arr` is a 2D array, normalize values across each row by dividing by the row sum.
    If you've already calculated row sums, you can pass those in, otherwise they will be computed.
    """
    if row_sums is None:
        row_sums = arr.sum(axis=1, dtype=dtype)
    # We do a maximum(1, ...) here to protect against div-by-zero:
    # if we assume `arr` is strictly non-negative and if a row-sum is zero,
    # then every entry in the row is zero therefore dividing by 1 is fine.
    return arr / np.maximum(1, row_sums[:, np.newaxis])  # type: ignore
    # numpy's types are garbage


RADIUS_MI = 3959.87433  # radius of earth in mi


def pairwise_haversine(longitudes: NDArray[np.float64], latitudes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the distances in miles between all pairs of coordinates."""
    # https://www.themathdoctors.org/distances-on-earth-2-the-haversine-formula
    lng = np.radians(longitudes)
    lat = np.radians(latitudes)
    dlng = lng[:, np.newaxis] - lng[np.newaxis, :]
    dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
    cos_lat = np.cos(lat)

    a = np.sin(dlat / 2.0) ** 2 \
        + (cos_lat[:, np.newaxis] * cos_lat[np.newaxis, :]) \
        * np.sin(dlng / 2.0) ** 2
    return 2 * RADIUS_MI * np.arcsin(np.sqrt(a))


def top(size: int, arr: NDArray) -> NDIndices:
    """
    Find the top `size` elements in `arr` and return their indices.
    Assumes the array is flat and the kind of thing that can be order-compared.
    """
    return np.argpartition(arr, -size)[-size:]


def bottom(size: int, arr: NDArray) -> NDIndices:
    """
    Find the bottom `size` elements in `arr` and return their indices.
    Assumes the array is flat and the kind of thing that can be order-compared.
    """
    return np.argpartition(arr, size)[:size]


def is_square(arr: NDArray) -> bool:
    """Is this numpy array 2 dimensions and square in shape?"""
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1]


_duration_regex = re.compile(r"^([0-9]+)([dwmy])$", re.IGNORECASE)


class NumpyTypeError(Exception):
    """Describes an error checking the type or shape of a numpy array."""


DT = TypeVar('DT', bound=np.generic)
"""A numpy dtype."""

DTLike = type[DT]
"""(Some) of the things that can be coerced as a numpy dtype."""


def check_ndarray(
    value: Any,
    dtype: DTLike[DT] | list[DTLike[DT]] | None = None,
    shape: tuple[int, ...] | list[tuple[int, ...]] | None = None,
    dimensions: int | list[int] | None = None,
) -> TypeGuard[NDArray[DT]]:
    """
    Checks that a value is a numpy array of the given dtype and shape.
    (If you pass a list of dtypes or shapes, they will be matched as though combined with an "or".)
    Type-guards if true, raises a NumpyTypeError if false.
    """
    if not isinstance(value, np.ndarray):
        raise NumpyTypeError("Not a numpy array.")
    if shape is not None:
        shape = as_list(shape)
        if not value.shape in shape:
            msg = f"Not a numpy shape match: got {value.shape}, expected {shape}"
            raise NumpyTypeError(msg)
    if dtype is not None:
        npdtypes = [np.dtype(x) for x in as_list(dtype)]
        is_subtype = map(lambda x: np.issubdtype(value.dtype, x), npdtypes)
        if not any(is_subtype):
            msg = f"Not a numpy dtype match; got {value.dtype}, required {npdtypes}"
            raise NumpyTypeError(msg)
    if dimensions is not None:
        dimensions = as_list(dimensions)
        if not len(value.shape) in dimensions:
            msg = f"Not a numpy dimensional match: got {len(value.shape)} dimensions, expected {dimensions}"
            raise NumpyTypeError(msg)
    return True


# custom pydantic types


class Duration(BaseModel):
    """Pydantic model describing a duration; serializes to/from a string representation."""
    # NOTE: the JSON schema for this isn't quite right but fixing that seems non-trivial.
    # Since we're not using JSON schema yet, a task for another day.
    count: int
    unit: Literal['d', 'w', 'm', 'y']

    def to_relativedelta(self) -> relativedelta:
        match self.unit:
            case "d":
                return relativedelta(days=self.count)
            case "w":
                return relativedelta(weeks=self.count)
            case "m":
                return relativedelta(months=self.count)
            case "y":
                return relativedelta(years=self.count)

    def __str__(self) -> str:
        return f"{self.count}{self.unit}"

    @model_serializer
    def _serialize(self) -> str:
        return str(self)

    @model_validator(mode='before')
    @classmethod
    def _validator(cls, value: Any) -> Any:
        if isinstance(value, str):
            match = _duration_regex.search(value)
            if not match:
                raise ValueError(
                    "not a valid duration (e.g., '100d' for 100 days)")
            else:
                count, unit = match.groups()
                return {'count': count, 'unit': unit}
        else:
            return value


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
        raise Exception("Code does not define a valid function")
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


# ImmutableNamespace


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
        return object.__getattribute__(self, __name)

    def __getattr__(self, __name: str) -> Any:
        data = object.__getattribute__(self, '_data')
        if __name not in data:
            __cls = self.__class__.__name__
            raise AttributeError(f"{__cls} object has no attribute '{__name}'")
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


def ns(data: dict[str, Any]) -> ImmutableNamespace:
    return ImmutableNamespace(data)
