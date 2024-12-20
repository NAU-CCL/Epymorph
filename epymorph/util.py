"""epymorph general utility functions and classes."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cache, wraps
from math import floor
from re import compile as re_compile
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    OrderedDict,
    ParamSpec,
    Self,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import deprecated

acceptable_name = re_compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
"""A pattern that matches generally acceptable names for use across epymorph."""


def normalize_str(value: str) -> str:
    """Normalize a string for permissive search."""
    return value.strip().lower()


def normalize_list(values: list[str]) -> list[str]:
    """Normalize a list of strings for permissive search."""
    return list(map(normalize_str, values))


class AnsiColor(Enum):
    # Standard Colors
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    # Bright Colors
    BRIGHT_BLACK = 90
    BRIGHT_RED = 91
    BRIGHT_GREEN = 92
    BRIGHT_YELLOW = 93
    BRIGHT_BLUE = 94
    BRIGHT_MAGENTA = 95
    BRIGHT_CYAN = 96
    BRIGHT_WHITE = 97


class AnsiStyle(Enum):
    RESET = 0  # Reset all styles
    BOLD = 1  # Bold text
    DIM = 2  # Dim text
    ITALIC = 3  # Italic text
    UNDERLINE = 4  # Underlined text
    STRIKETHROUGH = 9  # Strikethrough text


def ansi_stylize(
    text: str,
    color: AnsiColor | None = AnsiColor.WHITE,
    style: AnsiStyle | None = None,
) -> str:
    """Uses ANSI escape codes to stylize a given text, which may not work everywhere.
    Any applied color or style are reset after the text.

    Parameters
    ----------
    color : AnsiColor, optional
        The text color to apply.
    style : AnsiStyle, optional
        The text style to apply.
    """
    # ANSI reference: https://en.wikipedia.org/wiki/ANSI_escape_code
    codes = [str(x.value) for x in (color, style) if x is not None]
    if len(codes) == 0:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[{AnsiStyle.RESET.value}m"


# function utilities


T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")


def identity(x: T) -> T:
    """A function which just returns the argument it is called with."""
    return x


def constant(x: T) -> Callable[..., T]:
    """
    A function which returns a constant value,
    regardless of what arguments its called with.
    """
    return lambda *_: x


def noop():
    """A function which does nothing."""


def call_all(*fs: Callable[[], Any]) -> None:
    """Given a list of no-arg functions, call all of the functions and return None."""
    for f in fs:
        f()


P = ParamSpec("P")


def cache_transparent(func: Callable[P, T]) -> Callable[P, T]:
    """Decorates a function with functools.cache but maintains its signature."""
    # cache() transforms the function into an _lru_cache_wrapper,
    # which is technically correct but mostly unhelpful.
    # Generally we don't want the user to know or care that a function is cached.
    return wraps(func)(cache(func))  # type: ignore


# collection utilities


def index_where(it: Iterable[T], predicate: Callable[[T], bool]) -> int:
    """
    Find the first index of `it` where `predicate` evaluates to True.
    Return -1 if no such value exists.
    """
    for i, x in enumerate(it):
        if predicate(x):
            return i
    return -1


def index_of(it: Iterable[T], item: T) -> int:
    """
    Find the first index of `it` where `item` evaluates as equal.
    Return -1 if no such value exists.
    """
    for i, x in enumerate(it):
        if x == item:
            return i
    return -1


def iterator_length(it: Iterable[Any]) -> int:
    """
    Count the number of items in the given iterator.
    Warning: this consumes the iterator!
    It also never terminates if the iterator is infinite.
    """
    length = 0
    for _ in it:
        length += 1
    return length


def list_not_none(it: Iterable[T | None]) -> list[T]:
    """Convert an iterable to a list, skipping any entries that are None."""
    return [x for x in it if x is not None]


def are_unique(xs: Iterable[T]) -> bool:
    """Returns True if all items in the iterable are unique."""
    xset = set[T]()
    for x in xs:
        if x in xset:
            return False
        xset.add(x)
    return True


@overload
def are_instances(xs: list[Any], of_type: type[T]) -> TypeGuard[list[T]]: ...
@overload
def are_instances(xs: tuple[Any], of_type: type[T]) -> TypeGuard[tuple[T]]: ...


def are_instances(
    xs: list[Any] | tuple[Any], of_type: type[T]
) -> TypeGuard[list[T] | tuple[T]]:
    """
    TypeGuards a collection to check that all items are
    instances of the given type (`of_type`).
    """
    # NOTE: TypeVars can't be generic so we can't do TypeGuard[C[T]] :(
    # Thus this only supports the types of collections we specify explicitly.
    return all(isinstance(x, of_type) for x in xs)


def filter_unique(xs: Iterable[T]) -> list[T]:
    """
    Convert an iterable to a list, keeping only the unique values and
    maintaining the order as first-seen.
    """
    xset = set[T]()
    ys = list[T]()
    for x in xs:
        if x not in xset:
            ys.append(x)
            xset.add(x)
    return ys


def filter_with_mask(
    xs: Iterable[A], predicate: Callable[[A], TypeGuard[B]]
) -> tuple[list[B], list[bool]]:
    """
    Filters the given iterable for items which match `predicate`, and also
    returns a boolean mask the same length as the iterable with the results of
    `predicate` for each item.
    """
    matched = list[B]()
    mask = list[bool]()
    for x in xs:
        is_match = predicate(x)
        mask.append(is_match)
        if is_match:
            matched.append(x)
    return matched, mask


def zip_list(xs: Iterable[A], ys: Iterable[B]) -> list[tuple[A, B]]:
    """Zip (strict) two iterables together as a list."""
    return list(zip(xs, ys, strict=True))


K = TypeVar("K")
V = TypeVar("V")


class KeyValue(Generic[K, V], NamedTuple):
    """A generic named tuple for key/value pairs."""

    key: K
    value: V


def as_sorted_dict(x: dict[K, V]) -> OrderedDict[K, V]:
    """Returns a sorted OrderedDict of the given dict."""
    return OrderedDict(sorted(x.items()))


def map_values(f: Callable[[A], B], xs: Mapping[K, A]) -> dict[K, B]:
    """Maps the values of a Mapping into a dict by applying the given function."""
    return {k: f(v) for k, v in xs.items()}


def dict_map(m: Mapping[A, B], xs: Iterable[A]) -> list[B]:
    return [m[x] for x in xs]


class MemoDict(dict[K, V]):
    """
    A dict implementation which will call a factory function when the user attempts to
    access a key which is currently not in the dict.

    This varies slightly from `defaultdict`, which uses a factory function without the
    ability to pass the requested key.
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


N = TypeVar("N", bound=np.number)

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
        arr = np.pad(arr, pad_width=(0, pad), mode="constant", constant_values=0)
    return arr.reshape((-1, n)).sum(axis=1, dtype=dtype)


def normalize(arr: NDArray[N]) -> NDArray[N]:
    """
    Normalize the values in an array by subtracting the min and dividing by the range.
    """
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)


def row_normalize(
    arr: NDArray[N], row_sums: NDArray[N] | None = None, dtype: DTypeLike = None
) -> NDArray[N]:
    """
    Assuming `arr` is a 2D array, normalize values across each row by dividing
    by the row sum. If you've already calculated row sums, you can pass those in,
    otherwise they will be computed.
    """
    if row_sums is None:
        row_sums = arr.sum(axis=1, dtype=dtype)
    # We do a maximum(1, ...) here to protect against div-by-zero:
    # if we assume `arr` is strictly non-negative and if a row-sum is zero,
    # then every entry in the row is zero therefore dividing by 1 is fine.
    return arr / np.maximum(1, row_sums[:, np.newaxis])  # type: ignore
    # numpy's types are garbage


def prefix(length: int) -> Callable[[NDArray[np.str_]], NDArray[np.str_]]:
    """
    A vectorized operation to return the prefix of each value in an NDArray of strings.
    """
    return np.vectorize(lambda x: x[0:length], otypes=[np.str_])


def mask(length: int, selection: slice | list[int]) -> NDArray[np.bool_]:
    """
    Creates a boolean mask of a given `length` where all elements identified
    by `selection` are True and all others are False. The selection can be
    a slice or a list of indices.
    """
    mask = np.zeros(shape=length, dtype=np.bool_)
    mask[selection] = True
    return mask


# values from: https://www.themathdoctors.org/distances-on-earth-2-the-haversine-formula
_EARTH_RADIUS = {
    "miles": 3963.1906,
    "kilometers": 6378.1370,
}


def pairwise_haversine(
    coordinates: NDArray | tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    units: Literal["miles", "kilometers"] = "miles",
    radius: float | None = None,
) -> NDArray[np.float64]:
    """Compute the distances between all pairs of coordinates.

    Parameters
    ----------
    coordinates : NDArray | tuple[NDArray, NDArray]
        The coordinates, given in one of two forms: either a structured numpy array
        with dtype `[("longitude", np.float64), ("latitude", np.float64)]` or a tuple
        of two numpy arrays, the first containing longitudes and the second latitudes.
        The coordinates must be given in degrees.
    units : Literal["miles", "kilometers"] = "miles",
        The units of distance to use for the result, unless radius is given.
    radius : float, optional
        The radius of the Earth to use in calculating the results. If not given,
        we will use an appropriate value for the given `units`.
        Since the value of radius implies the distance units being used, if you
        specify `radius` the value of `units` is ignored.

    Returns
    -------
    NDArray[np.float64] :
        An NxN array of distances where N is the number of coordinates given,
        representing the distance between each pair of coordinates. The output
        maintains the same ordering of coordinates as the input.

    Raises
    ------
    ValueError :
        if coordinates are not given in an expected format
    """
    # https://www.themathdoctors.org/distances-on-earth-2-the-haversine-formula
    if isinstance(coordinates, np.ndarray) and coordinates.dtype == np.dtype(
        [("longitude", np.float64), ("latitude", np.float64)]
    ):
        lng = coordinates["longitude"]
        lat = coordinates["latitude"]
    elif isinstance(coordinates, tuple) and len(coordinates) == 2:
        lng = coordinates[0]
        lat = coordinates[1]
    else:
        err = "Unable to interpret the given `coordinates`."
        raise ValueError(err)

    lngrad = np.radians(lng)
    latrad = np.radians(lat)
    dlng = lngrad[:, np.newaxis] - lngrad[np.newaxis, :]
    dlat = latrad[:, np.newaxis] - latrad[np.newaxis, :]
    cos_lat = np.cos(latrad)

    a = (
        np.sin(dlat / 2.0) ** 2
        + (cos_lat[:, np.newaxis] * cos_lat[np.newaxis, :]) * np.sin(dlng / 2.0) ** 2
    )
    r = radius if radius is not None else _EARTH_RADIUS[units]
    return 2 * r * np.arcsin(np.sqrt(a))


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


def shape_matches(arr: NDArray, expected: tuple[int | Literal["?"], ...]) -> bool:
    """
    Does the shape of the given array match this expression?
    Shape expressions are a tuple where each dimension is either an integer
    or a '?' character to signify any length is allowed.
    """
    if len(arr.shape) != len(expected):
        return False
    for actual, exp in zip(arr.shape, expected):
        if exp == "?":
            continue
        if exp != actual:
            return False
    return True


class NumpyTypeError(Exception):
    """Describes an error checking the type or shape of a numpy array."""


def dtype_name(d: np.dtype) -> str:
    """Tries to return the most-human-readable name for a numpy dtype."""
    if np.issubdtype(d, np.str_):
        return "str_"
    if d.isbuiltin:
        return d.name
    return str(d)


# Matchers


T_contra = TypeVar("T_contra", contravariant=True)


class Matcher(Generic[T_contra], ABC):
    """
    A generic matcher. Returns True if a match, False otherwise.
    """

    # Note: Matchers are contravariant: you can substitute a Matcher of a broader type
    # when something asks for a Matcher of a more specific type.
    # For example, a Matcher[Any] can be provided in place of a Matcher[str].

    @abstractmethod
    def expected(self) -> str:
        """Describes what the expected value is."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, value: Any) -> bool:
        raise NotImplementedError


class MatchAny(Matcher[Any]):
    """Always matches (returns True)."""

    def expected(self) -> str:
        return "any value"

    def __call__(self, _value: Any) -> bool:
        return True


class MatchEqual(Matcher[T_contra]):
    """Matches a specific value by checking for equality (==)."""

    _acceptable: T_contra

    def __init__(self, acceptable: T_contra):
        self._acceptable = acceptable

    def expected(self) -> str:
        return str(self._acceptable)

    def __call__(self, value: Any) -> bool:
        return value == self._acceptable


class MatchAnyIn(Matcher[T_contra]):
    """Matches for presence in a list of values (in)."""

    _acceptable: list[T_contra]

    def __init__(self, acceptable: list[T_contra]):
        self._acceptable = acceptable

    def expected(self) -> str:
        return f"one of [{', '.join((str(x) for x in self._acceptable))}]"

    def __call__(self, value: T_contra) -> bool:
        return value in self._acceptable


class MatchDType(Matcher[DTypeLike]):
    """Matches one or more numpy dtypes using `np.issubdtype()`."""

    _acceptable: list[np.dtype]

    def __init__(self, *acceptable: DTypeLike):
        if len(acceptable) == 0:
            raise ValueError("Cannot match against no dtypes.")
        self._acceptable = [np.dtype(x) for x in acceptable]

    def expected(self) -> str:
        if len(self._acceptable) == 1:
            return dtype_name(self._acceptable[0])
        else:
            return f"one of [{', '.join((dtype_name(x) for x in self._acceptable))}]"

    def __call__(self, value: DTypeLike) -> bool:
        return any((np.issubdtype(value, x) for x in self._acceptable))


class MatchDTypeCast(Matcher[DTypeLike]):
    """Matches one or more numpy dtypes using `np.can_cast(casting='safe')`."""

    _acceptable: list[np.dtype]

    def __init__(self, *acceptable: DTypeLike):
        if len(acceptable) == 0:
            raise ValueError("Cannot match against no dtypes.")
        self._acceptable = [np.dtype(x) for x in acceptable]

    def expected(self) -> str:
        if len(self._acceptable) == 1:
            return dtype_name(self._acceptable[0])
        else:
            return f"one of [{', '.join((dtype_name(x) for x in self._acceptable))}]"

    def __call__(self, value: DTypeLike) -> bool:
        return any((np.can_cast(value, x, casting="safe") for x in self._acceptable))


class MatchShapeLiteral(Matcher[NDArray]):
    """
    Matches a numpy array shape to a known literal value.
    (For matching relative to simulation dimensions, you want DataShapeMatcher.)
    """

    _acceptable: tuple[int, ...]

    def __init__(self, acceptable: tuple[int, ...]):
        self._acceptable = acceptable

    def expected(self) -> str:
        """Describes what the expected value is."""
        return str(self._acceptable)

    def __call__(self, value: NDArray) -> bool:
        return self._acceptable == value.shape


class MatchDimensions(Matcher[NDArray]):
    """
    Matches a numpy array purely on the number of dimensions.
    """

    _acceptable: int

    def __init__(self, acceptable: int):
        if acceptable < 0:
            err = "Dimensions must be greater than or equal to zero."
            raise ValueError(err)
        self._acceptable = acceptable

    def expected(self) -> str:
        """Describes what the expected value is."""
        return f"{self._acceptable}-dimensional array"

    def __call__(self, value: NDArray) -> bool:
        return self._acceptable == value.ndim


@dataclass(frozen=True)
class _Matchers:
    """Convenience constructors for various matchers."""

    any = MatchAny()
    """A matcher that matches any value. (Singleton instance of MatchAny.)"""

    def equal(self, value: T) -> Matcher[T]:
        """Creates a MatchEqual instance."""
        return MatchEqual(value)

    def any_in(self, values: list[T]) -> Matcher[T]:
        """Creates a MatchAnyIn instance."""
        return MatchAnyIn(values)

    def dtype(self, *dtypes: DTypeLike) -> Matcher[DTypeLike]:
        """Creates a MatchDType instance."""
        return MatchDType(*dtypes)

    def dtype_cast(self, *dtypes: DTypeLike) -> Matcher[DTypeLike]:
        """Creates a MatchDTypeCast instance."""
        return MatchDTypeCast(*dtypes)

    def shape_literal(self, shape: tuple[int, ...]) -> Matcher[NDArray]:
        """Creates a MatchShapeLiteral instance."""
        return MatchShapeLiteral(shape)

    def dimensions(self, dimensions: int) -> Matcher[NDArray]:
        """Creates a MatchDimensions instance."""
        return MatchDimensions(dimensions)


match = _Matchers()
"""Convenience constructors for various matchers."""


def check_ndarray(
    value: Any,
    *,
    dtype: Matcher[DTypeLike] = MatchAny(),
    shape: Matcher[NDArray] = MatchAny(),
) -> None:
    """
    Checks that a value is a numpy array matching the given dtype and shape Matchers.
    Raises a NumpyTypeError if a check doesn't pass.
    """
    if value is None:
        raise NumpyTypeError("Value is None.")

    if not isinstance(value, np.ndarray):
        raise NumpyTypeError("Not a numpy array.")

    if not dtype(value.dtype):
        msg = (
            "Not a numpy dtype match; "
            f"got {dtype_name(value.dtype)}, required {dtype.expected()}"
        )
        raise NumpyTypeError(msg)

    if not shape(value):
        msg = f"Not a numpy shape match: got {value.shape}, expected {shape.expected()}"
        raise NumpyTypeError(msg)


# console decorations


def progress(percent: float, length: int = 20) -> str:
    """Creates a progress bar string."""
    if length < 1:
        raise ValueError("progress bar length cannot be less than 1")
    p = max(0.0, min(percent, 1.0))
    n = floor(length * p)
    bar = ("#" * n) + (" " * (length - n))
    return f"|{bar}| {(100 * p):.0f}% "


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

    @property
    def has_subscribers(self) -> bool:
        """True if at least one listener is subscribed to this event."""
        return len(self._subscribers) > 0


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
    Manage a subscription context, where all subscriptions added through the returned
    Subscriber will be automatically unsubscribed when the context closes.
    """
    sub = Subscriber()
    try:
        yield sub
    finally:
        sub.unsubscribe()


# singletons


class Singleton(type):
    """A metaclass for classes you want to treat as singletons."""

    _instances: dict[type["Singleton"], "Singleton"] = {}

    def __call__(cls: type["Singleton"], *args: Any, **kwargs: Any) -> "Singleton":
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# string builders


class StringBuilder:
    _lines: list[str]
    _indent: str

    def __init__(self, indent: str = ""):
        self._lines = []
        self._indent = indent

    def line(self, line: str = "") -> Self:
        self._lines.append(line)
        return self

    def line_if(self, condition: bool, line: str = "") -> Self:
        if condition:
            self._lines.append(line)
        return self

    def lines(self, lines: Iterable[str]) -> Self:
        self._lines.extend(lines)
        return self

    @contextmanager
    def block(
        self,
        indent: str = "    ",
        *,
        opener: str | None = None,
        closer: str | None = None,
    ) -> Generator["StringBuilder", None, None]:
        if opener is not None:
            # opener is printed at the parent's indent level
            self.line(opener)

        new_indent = f"{self._indent}{indent}"
        s = StringBuilder(new_indent)
        yield s
        self._lines.extend((f"{new_indent}{line}" for line in s.to_lines()))

        if closer is not None:
            # closer is printed at the parent's indent level
            self.line(closer)

    def build(self) -> str:
        return "\n".join(self._lines)

    def to_lines(self) -> Iterable[str]:
        return self._lines


@contextmanager
def string_builder(
    indent: str = "", *, opener: str | None = None, closer: str | None = None
) -> Generator[StringBuilder, None, None]:
    s = StringBuilder(indent)
    if opener is not None:
        s.line(opener)
    yield s
    if closer is not None:
        s.line(closer)
