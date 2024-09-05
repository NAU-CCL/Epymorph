# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock

import numpy as np

from epymorph import util
from epymorph.data_shape import DataShapeMatcher, Shapes, SimDimensions
from epymorph.util import match as m


class TestUtil(unittest.TestCase):
    def test_identity(self):
        tests = [1, "hey", [1, 2, 3], {"foo": "bar"}]
        for t in tests:
            self.assertEqual(t, util.identity(t))

    def test_stutter(self):
        act = list(util.stutter(["a", "b", "c"], 3))
        exp = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
        self.assertEqual(act, exp)

    def test_stridesum(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

        act1 = util.stridesum(arr, 2)
        exp1 = np.array([3, 7, 11, 15, 19])
        self.assertTrue(all(np.equal(act1, exp1)))

        act2 = util.stridesum(arr, 5)
        exp2 = np.array([15, 40])
        self.assertTrue(all(np.equal(act2, exp2)))

        act3 = util.stridesum(arr, 3)
        exp3 = np.array([6, 15, 24, 10])
        self.assertTrue(all(np.equal(act3, exp3)))

    def test_filter_unique(self):
        act = util.filter_unique(["a", "b", "b", "c", "a"])
        exp = ["a", "b", "c"]
        self.assertListEqual(act, exp)

    def test_list_not_none(self):
        act = util.list_not_none(["a", None, "b", None, None, "c", None])
        exp = ["a", "b", "c"]
        self.assertListEqual(act, exp)

    def test_check_ndarray_01(self):
        # None of these should raise NumpyTypeError
        arr = np.array([1, 2, 3], dtype=np.int64)

        dim = Mock(SimDimensions)
        dim.nodes = 3
        dim.days = 10

        util.check_ndarray(arr)
        util.check_ndarray(arr, dtype=m.dtype(np.int64))
        util.check_ndarray(arr, shape=DataShapeMatcher(Shapes.N, dim, True))
        util.check_ndarray(
            arr, dtype=m.dtype(np.int64), shape=DataShapeMatcher(Shapes.N, dim, True)
        )
        util.check_ndarray(
            arr,
            dtype=m.dtype(np.int64, np.float64),
            shape=DataShapeMatcher(Shapes.N, dim, True),
        )
        util.check_ndarray(
            arr,
            dtype=m.dtype(np.float64, np.int64),
            shape=DataShapeMatcher(Shapes.N, dim, False),
        )
        util.check_ndarray(
            arr,
            dtype=m.dtype(np.int64, np.float64),
            shape=DataShapeMatcher(Shapes.TxN, dim, True),
        )

    def test_check_ndarray_02(self):
        # Raises exception for anything that's not a numpy array
        with self.assertRaises(util.NumpyTypeError):
            util.check_ndarray(None)
        with self.assertRaises(util.NumpyTypeError):
            util.check_ndarray(1)
        with self.assertRaises(util.NumpyTypeError):
            util.check_ndarray([1, 2, 3])
        with self.assertRaises(util.NumpyTypeError):
            util.check_ndarray("foofaraw")

    def test_check_ndarray_03(self):
        arr = np.arange(12).reshape((3, 4))

        # Doesn't raise...
        dim1 = Mock(SimDimensions)
        dim1.nodes = 4
        dim1.days = 3
        util.check_ndarray(arr, shape=DataShapeMatcher(Shapes.TxN, dim1, True))

        # Does raise...
        with self.assertRaises(util.NumpyTypeError):
            dim2 = Mock(SimDimensions)
            dim2.nodes = 3
            dim2.days = 4
            util.check_ndarray(arr, shape=DataShapeMatcher(Shapes.TxN, dim2, True))
        with self.assertRaises(util.NumpyTypeError):
            util.check_ndarray(arr, dtype=m.dtype(np.str_))
