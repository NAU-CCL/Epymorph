import re
from typing import Any, Callable, Iterable, TypeVar

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
