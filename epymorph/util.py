"""epymorph general utility functions and classes."""

from abc import ABC, abstractmethod
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
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
    NamedTuple,
    ParamSpec,
    Protocol,
    Self,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import DTypeLike, NDArray

acceptable_name = re_compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
"""A pattern that matches generally acceptable names for use across epymorph."""


def normalize_str(value: str) -> str:
    """Normalize a string for permissive search."""
    return value.strip().lower()


def normalize_list(values: list[str]) -> list[str]:
    """Normalize a list of strings for permissive search."""
    return list(map(normalize_str, values))


class ANSIColor(Enum):
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


class ANSIStyle(Enum):
    RESET = 0  # Reset all styles
    BOLD = 1  # Bold text
    DIM = 2  # Dim text
    ITALIC = 3  # Italic text
    UNDERLINE = 4  # Underlined text
    STRIKETHROUGH = 9  # Strikethrough text


def ansi_stylize(
    text: str,
    color: ANSIColor | None = ANSIColor.WHITE,
    style: ANSIStyle | None = None,
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
    return f"\033[{';'.join(codes)}m{text}\033[{ANSIStyle.RESET.value}m"


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


K_co = TypeVar("K_co", covariant=True)
V_co = TypeVar("V_co", covariant=True)


class CovariantMapping(Protocol[K_co, V_co]):
    """A type for covariant mappings, which restricts usage
    to only those methods which are safe under covariance.
    For many use-cases these limitations are acceptable and
    wind up simplifying type expression."""

    @abstractmethod
    def keys(self) -> KeysView[K_co]: ...

    @abstractmethod
    def items(self) -> ItemsView[K_co, V_co]: ...

    @abstractmethod
    def values(self) -> ValuesView[V_co]: ...

    @abstractmethod
    def __len__(self) -> int: ...


def as_dict(mapping: CovariantMapping[K, V]) -> dict[K, V]:
    return {k: v for k, v in mapping.items()}


def map_values(f: Callable[[A], B], xs: Mapping[K, A]) -> dict[K, B]:
    """Maps the values of a Mapping into a dict by applying the given function."""
    return {k: f(v) for k, v in xs.items()}


# numpy utilities


N = TypeVar("N", bound=np.number)

NDIndices = NDArray[np.intp]


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


def is_numeric(arr: NDArray) -> TypeGuard[NDArray[np.integer | np.floating]]:
    """Is this numpy array a numeric (non-complex) type?"""
    return np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)


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


####################
# DATE/VALUE TYPES #
####################


DataT = TypeVar("DataT", bound=np.generic)

DateValueType = np.void
"""
The numpy dtype used for structured arrays of date and value.
numpy doesn't let us type these very explicitly, so this alias
is used to express intention, but know it comes with few guarantees.

So basically it implies: `[("date", "datetime64[D]"), ("value", DataT)]`
"""


def date_value_dtype(value_dtype: DTypeLike) -> np.dtype:
    return np.dtype([("date", "datetime64[D]"), ("value", value_dtype)])


def is_date_value_dtype(
    dtype: np.dtype | type[np.generic],
    *,
    value_dtype: np.dtype | type[np.generic] | None = None,
) -> TypeGuard[DateValueType]:
    """
    Check if the given dtype is structured with 'date' and 'value' fields.

    Parameters
    ----------
    dtype :
        The dtype to check.
    value_dtype :
        The optional expected dtype of the 'value' field. If None (default),
        the dtype of the 'value' field is not checked.

    Returns
    -------
    :
        True if the dtype is date/value, false otherwise.

    See Also
    --------
    [epymorph.util.is_date_value_array][] for examples.
    """

    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if dtype.names != ("date", "value"):
        return False
    if dtype["date"] != np.dtype("datetime64[D]"):
        return False
    if value_dtype:
        if not isinstance(value_dtype, np.dtype):
            value_dtype = np.dtype(value_dtype)
        if dtype["value"] != value_dtype:
            return False
    return True


def is_date_value_array(
    array: NDArray,
    *,
    value_dtype: type[DataT] | None = None,
) -> TypeGuard[NDArray[DateValueType]]:
    """
    Check if the given array is a structured array with 'date' and 'value' fields.

    Parameters
    ----------
    array : NDArray
        The array to check.
    value_dtype : type[DataT], optional
        The optional expected dtype of the 'value' field. If None (default),
        the dtype of the 'value' field is not checked.

    Returns
    -------
    TypeGuard[NDArray[DateValueType]]
        True if the array is a date/value array, false otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> array = np.array([('2021-01-01', 10), ('2021-01-02', 20)],
                         dtype=[('date', 'datetime64[D]'), ('value', np.int64)])
    >>> is_date_value_array(array)
    True
    >>> is_date_value_array(array, value_dtype=np.int64)
    True
    """
    return is_date_value_dtype(array.dtype, value_dtype=value_dtype)


def to_date_value_array(
    dates: NDArray[np.datetime64],
    values: NDArray[DataT],
) -> NDArray[DateValueType]:
    """
    Combine separate arrays of dates and values to create a structured date/value array
    with named fields. The given dates will be broadcast against the given values,
    so they must have the same first-dimension.

    Parameters
    ----------
    dates : NDArray[np.datetime64]
        An 1D array of dates.
    values : NDArray[DataT]
        An array of values corresponding to the dates,
        the first dimension of this array must be equal to the number of dates.

    Returns
    -------
    NDArray[DateValueType]
        A structured array with fields 'date' and 'value'.

    Raises
    ------
    ValueError
        If the dimensionality of the arrays is not compatible.

    Examples
    --------
    >>> import numpy as np
    >>> dates = np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]')
    >>> values = np.array([[10, 20], [30, 40]])
    >>> to_date_value_array(dates, values)
    array(
        [
            [('2021-01-01', 10), ('2021-01-01', 20)],
            [('2021-01-02', 30), ('2021-01-02', 40)],
        ],
        dtype=[('date', 'datetime64[D]'), ('value', '<i8')],
    )
    """
    if dates.ndim != 1:
        raise ValueError("`dates` must be a 1D array.")
    if dates.shape[0] != values.shape[0]:
        err = "The first axis of `dates` and `values` must be the same length."
        raise ValueError(err)
    value_dtype = np.dtype(values.dtype).type
    result = np.empty(
        values.shape,
        dtype=[("date", "datetime64[D]"), ("value", value_dtype)],
    )
    result["date"] = dates.reshape(dates.shape + (1,) * (values.ndim - 1))
    result["value"] = values
    return result


def extract_date_value(
    date_values: NDArray[DateValueType],
    value_dtype: type[DataT] | None = None,
) -> tuple[NDArray[np.datetime64], NDArray[DataT]]:
    """
    Extract separate arrays of dates and values from a structured date/value array.

    Parameters
    ----------
    date_values : NDArray[DateValueType]
        A structured array with fields 'date' and 'value'.
    value_dtype : type[DataT], optional
        The expected dtype of the 'value' field. If None (default),
        the dtype of the 'value' field is not checked.

    Returns
    -------
    tuple[NDArray[np.datetime64], NDArray[DataT]]
        A tuple containing two arrays: dates and values.

    Raises
    ------
    ValueError
        If `value_dtype` is provided and the dtype of the 'value' field does not match.

    Examples
    --------
    >>> import numpy as np
    >>> date_values = np.array([('2021-01-01', 10), ('2021-01-02', 20)],
                               dtype=[('date', 'datetime64[D]'), ('value', np.int64)])
    >>> extract_date_value(date_values)
    (array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]'),
     array([10, 20]))
    """
    # For dates: get the first axis and then the 0th element of all other axes
    # in other words, `[:, 0, 0, 0]` for however many 0s are necessary
    date_slice = np.index_exp[:] + np.index_exp[0] * (date_values.ndim - 1)
    dates = np.ma.getdata(date_values["date"][date_slice])
    values = date_values["value"]
    if value_dtype and np.dtype(values.dtype) != np.dtype(value_dtype):
        err = "Date/value array's values did not match expected dtype."
        raise ValueError(err)
    return dates, values


# Matchers


MatcherValueType_contra = TypeVar("MatcherValueType_contra", contravariant=True)
"""The type of a value matcher by a `Matcher`."""


class Matcher(Generic[MatcherValueType_contra], ABC):
    """
    The root class for a family of predicates on values; `Matcher` child classes
    are written to match specific types of values, and encode conditions against which
    values are tested. Once you have a concrete `Matcher` instance, you use it like a
    function (it implements `__call__` semantics) to test a value.
    """

    # Note: Matchers are contravariant: you can substitute a Matcher of a broader type
    # when something asks for a Matcher of a more specific type.
    # For example, a Matcher[Any] can be provided in place of a Matcher[str].

    @abstractmethod
    def expected(self) -> str:
        """
        The description of what this matcher's expected valid value(s) are.

        Returns
        -------
        :
            The value(s) we expect to match, as a string.
        """

    @abstractmethod
    def __call__(self, value: Any) -> bool:
        """
        Test whether or not the given value matches this matcher.

        Parameters
        ----------
        value :
            The value to match.

        Returns
        -------
        :
            Whether or not the match is successful.
        """


class MatchAny(Matcher[Any]):
    """Always matches (returns True)."""

    def expected(self) -> str:
        return "any value"

    def __call__(self, _value: Any) -> bool:
        return True


class MatchEqual(Matcher[MatcherValueType_contra]):
    """Matches a specific value by checking for equality (==)."""

    _acceptable: MatcherValueType_contra

    def __init__(self, acceptable: MatcherValueType_contra):
        self._acceptable = acceptable

    def expected(self) -> str:
        return str(self._acceptable)

    def __call__(self, value: Any) -> bool:
        return value == self._acceptable


class MatchAnyIn(Matcher[MatcherValueType_contra]):
    """Matches for presence in a list of values (in)."""

    _acceptable: list[MatcherValueType_contra]

    def __init__(self, acceptable: list[MatcherValueType_contra]):
        self._acceptable = acceptable

    def expected(self) -> str:
        return f"one of [{', '.join((str(x) for x in self._acceptable))}]"

    def __call__(self, value: MatcherValueType_contra) -> bool:
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

SingletonT = TypeVar("SingletonT", bound="Singleton")


class Singleton(type):
    """A metaclass for classes you want to treat as singletons."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
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
