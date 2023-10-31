"""Epymorph unit testing utilities."""
from unittest import TestCase

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _ndarray_signature(arr):
    return f"{type(arr).__name__}: dtype({arr.dtype}) shape{repr(arr.shape)}"


class EpymorphTestCase(TestCase):
    """A unittest TestCase extension providing some extra utility functions."""

    def assertNpEqual(self, a1: ArrayLike, a2: ArrayLike, msg: str | None = None) -> None:
        """Check that two numpy ArrayLikes are equal."""
        if not np.array_equal(a1, a2):
            if msg is None:
                if not isinstance(a1, np.ndarray):
                    a1 = np.asarray(a1)
                if not isinstance(a2, np.ndarray):
                    a2 = np.asarray(a2)
                sig1 = _ndarray_signature(a1)
                sig2 = _ndarray_signature(a2)
                msg = f"""\
arrays not equal
- a1: {sig1}
{str(a1)}
- a2: {sig2}
{str(a2)}"""
            self.fail(msg)
