import re
from typing import Any, Callable, Generic, Iterable, TypeVar

import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

Compartments = NDArray[np.int_]
Events = NDArray[np.int_]
DataDict = dict[str, Any]


T = TypeVar('T')


def identity(x: T) -> T:
    return x


def constant(x: T) -> Callable[..., T]:
    return lambda *_: x


def stutter(it: Iterable[T], times: int) -> Iterable[T]:
    """Make the iterable `it` repeat each item `times` times.
       (Unlike `itertools.repeat` which repeats whole sequences in order.)"""
    return (xs for x in it for xs in (x,) * times)


def stridesum(arr: NDArray[np.int_], n: int) -> NDArray[np.int_]:
    """Compute a new array by grouping every `n` rows and summing them together.
       `arr`'s length must be evenly divisible by `n`."""
    rows = len(arr)
    assert rows % n == 0, f"Cannot stridesum array of length {rows} by {n}."
    res = np.zeros(shape=rows // n, dtype=np.int_)
    for j in range(0, rows, n):
        sum = np.int_(0)
        for i in range(0, n):
            sum += arr[j + i]
        res[j // n] = sum
    return res


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


def progress(percent: float) -> str:
    """Creates a progress bar string."""
    p = 100 * max(0.0, min(percent, 1.0))
    n = int(p // 5)
    bar = ('#' * n) + (' ' * (20 - n))
    return f"|{bar}| {p:.0f}% "


T = TypeVar('T')


class Event(Generic[T]):
    subscribers: list[Callable[[T], None]]

    def __init__(self):
        self.subscribers = []

    def subscribe(self, sub: Callable[[T], None]) -> None:
        self.subscribers.append(sub)

    def publish(self, event: T) -> None:
        for subscriber in self.subscribers:
            subscriber(event)
