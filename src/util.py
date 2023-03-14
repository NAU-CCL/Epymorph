from typing import Iterable, TypeVar

import numpy as np
from numpy.typing import NDArray

Compartments = NDArray[np.int_]
Events = NDArray[np.int_]


T = TypeVar('T')


def identity(x):
    return x


def stutter(it: Iterable[T], times: int) -> Iterable[T]:
    """Make the iterable `it` repeat each item `times` times.
       (Unlike `itertools.repeat` which repeats whole sequences in order.)"""
    return (xs for x in it for xs in (x,) * times)


def stridesum(arr: NDArray[np.int_], n: int) -> NDArray[np.int_]:
    """Compute a new array by grouping every `n` rows and summing them together.
       `arr`'s length must be evenly divisible by `n`."""
    rows = len(arr)
    res = np.zeros(shape=rows // n, dtype=np.int_)
    for j in range(0, rows, n):
        sum = np.int_(0)
        for i in range(0, n):
            sum += arr[j + i]
        res[j // n] = sum
    return res


def is_square(arr: NDArray) -> bool:
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1]
