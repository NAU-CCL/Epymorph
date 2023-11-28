import ast
import re
from contextlib import contextmanager
from typing import (Any, Callable, Generator, Generic, Iterable, Literal,
                    OrderedDict, TypeGuard, TypeVar)

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import deprecated

# function utilities


T = TypeVar('T')


def identity(x: T) -> T:
    return x


def constant(x: T) -> Callable[..., T]:
    return lambda *_: x


def noop():
    """A function which does nothing."""


def call_all(*fs: Callable[[], Any]) -> None:
    """Given a list of no-arg functions, call all of the functions and return None."""
    for f in fs:
        f()


# collection utilities


def index_where(it: Iterable[T], predicate: Callable[[T], bool]) -> int:
    """Find the first index of `it` where `predicate` evaluates to True. Return -1 if no such value exists."""
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


class MemoDict(dict[K, V]):
    """
    A dict implementation which will call a factory function when the user attempts to access
    a key which is currently not in the dict.

    This varies slightly from `defaultdict`, which uses a factory function without the ability
    to pass the requested key.
    """

    _factory: Callable[[K], V]

    def __init__(self, factory: Callable[[K], V]):
        super().__init__()
        self._factory = factory

    def __missing__(self, key: K) -> V:
        value = self._factory(key)
        self[key] = value
        return value


# numpy utilities


N = TypeVar('N', bound=np.number)

NDIndices = NDArray[np.intp]


@deprecated("Don't use this; use np.repeat")
def stutter(it: Iterable[T], times: int) -> Iterable[T]:
    """Make the iterable `it` repeat each item `times` times.
       (Unlike `itertools.repeat` which repeats whole sequences in order.)"""
    return (xs for x in it for xs in (x,) * times)


@deprecated("Don't use this; use np.reshape and np.sum")
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


def shape_matches(arr: NDArray, expected: tuple[int | Literal['?'], ...]) -> bool:
    """
    Does the shape of the given array match this expression?
    Shape expressions are a tuple where each dimension is either an integer
    or a '?' character to signify any length is allowed.
    """
    if len(arr.shape) != len(expected):
        return False
    for actual, exp in zip(arr.shape, expected):
        if exp == '?':
            continue
        if exp != actual:
            return False
    return True


class NumpyTypeError(Exception):
    """Describes an error checking the type or shape of a numpy array."""


DT = TypeVar('DT', bound=np.generic)
"""A numpy dtype."""

DTLike = type[DT]
"""(Some) of the things that can be coerced as a numpy dtype."""


def dtype_name(d: np.dtype) -> str:
    """Tries to return the most-human-readable name for a numpy dtype."""
    return d.name if d.isbuiltin else str(d)


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
    if value is None:
        raise NumpyTypeError('Value is None.')
    if not isinstance(value, np.ndarray):
        raise NumpyTypeError('Not a numpy array.')
    if shape is not None:
        shape = as_list(shape)
        if not value.shape in shape:
            msg = f"Not a numpy shape match: got {value.shape}, expected {shape}"
            raise NumpyTypeError(msg)
    if dtype is not None:
        npdtypes = [np.dtype(x) for x in as_list(dtype)]
        is_subtype = map(lambda x: np.issubdtype(value.dtype, x), npdtypes)
        if not any(is_subtype):
            if len(npdtypes) == 1:
                dtype_names = dtype_name(npdtypes[0])
            else:
                dtype_names = f"one of ({', '.join(map(dtype_name, npdtypes))})"
            msg = f"Not a numpy dtype match; got {value.dtype}, required {dtype_names}"
            raise NumpyTypeError(msg)
    if dimensions is not None:
        dimensions = as_list(dimensions)
        if not len(value.shape) in dimensions:
            msg = f"Not a numpy dimensional match: got {len(value.shape)} dimensions, expected {dimensions}"
            raise NumpyTypeError(msg)
    return True


# console decorations


def progress(percent: float) -> str:
    """Creates a progress bar string."""
    p = 100 * max(0.0, min(percent, 1.0))
    n = int(p // 5)
    bar = ('#' * n) + (' ' * (20 - n))
    return f"|{bar}| {p:.0f}% "


# pub-sub events


class Event(Generic[T]):
    """A typed pub-sub event."""
    _subscribers: list[Callable[[T], None]]

    def __init__(self):
        self._subscribers = []

    def subscribe(self, sub: Callable[[T], None]) -> Callable[[], None]:
        """Subscribe a handler to this event. Returns an unsubscribe function."""
        self._subscribers.append(sub)

        def unsubscribe() -> None:
            self._subscribers.remove(sub)
        return unsubscribe

    def publish(self, event: T) -> None:
        """Publish an event occurrence to all current subscribers."""
        for subscriber in self._subscribers:
            subscriber(event)


class Subscriber:
    """
    Utility class to track a list of subscriptions for ease of unsubscription.
    Consider using this via the `subscriptions()` context.
    """

    _unsubscribers: list[Callable[[], None]]

    def __init__(self):
        self._unsubscribers = []

    def subscribe(self, event: Event[T], handler: Callable[[T], None]) -> None:
        """Subscribe through this Subscriber to the given event."""
        unsub = event.subscribe(handler)
        self._unsubscribers.append(unsub)

    def unsubscribe(self) -> None:
        """Unsubscribe from all of this Subscriber's subscriptions."""
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()


@contextmanager
def subscriptions() -> Generator[Subscriber, None, None]:
    """
    Manage a subscription context, where all subscriptions added through the returned Subscriber
    will be automatically unsubscribed when the context closes.
    """
    sub = Subscriber()
    yield sub
    sub.unsubscribe()


# AST function utilities


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


# ImmutableNamespace


class ImmutableNamespace:
    """A simple dot-accessible dictionary."""
    # TODO: is it possible to use vars(my_immutable_namespace) to get a mutable handle on this class' data?

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
