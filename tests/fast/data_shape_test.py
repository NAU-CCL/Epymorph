# ruff: noqa: PT009,PT027
import unittest
from functools import reduce
from typing import Any, TypeGuard, TypeVar

import numpy as np

from epymorph.data_shape import Dimensions, Shapes, parse_shape

_to_str = np.vectorize(str)

_dim = Dimensions.of(T=90, N=6, C=3, E=2)

T = TypeVar("T")


def shaped_arange(shape: tuple[int, ...]):
    return np.arange(reduce(lambda a, b: a * b, shape)).reshape(shape)


class DataShape(unittest.TestCase):
    def assert_not_none(self, val: T | None, msg: Any | None = None) -> TypeGuard[T]:
        self.assertIsNotNone(val, msg)
        return True

    def as_bool_asserts(self, test_fn):
        def ttt(*args, **kwargs):
            return self.assertTrue(test_fn(*args, **kwargs))

        def fff(*args, **kwargs):
            return self.assertFalse(test_fn(*args, **kwargs))

        return ttt, fff

    def test_scalar(self):
        ttt, fff = self.as_bool_asserts(lambda x: Shapes.Scalar.matches(_dim, x))

        ttt(np.asarray(1))
        ttt(np.asarray(3.14159))
        ttt(np.asarray("this is a string"))

        fff(np.array([1]))
        fff(np.array([1, 2, 3]))
        fff(np.arange(9).reshape((3, 3)))

    def test_time(self):
        ttt, fff = self.as_bool_asserts(lambda x: Shapes.T.matches(_dim, x))

        ttt(np.arange(90))
        ttt(np.arange(99))
        ttt(_to_str(np.arange(90)))

        ttt(np.asarray(42))

        fff(np.arange(6))
        fff(np.arange(90 * 2).reshape((90, 2)))

    def test_node(self):
        ttt, fff = self.as_bool_asserts(lambda x: Shapes.N.matches(_dim, x))

        ttt(np.arange(6))
        ttt(_to_str(np.arange(6)))

        ttt(np.asarray(42))

        fff(np.arange(5))
        fff(np.arange(7))
        fff(np.arange(90))
        fff(np.arange(6 * 6 * 6).reshape((6, 6, 6)))

    def test_time_and_node(self):
        ttt, fff = self.as_bool_asserts(lambda x: Shapes.TxN.matches(_dim, x))

        ttt(np.arange(90 * 6).reshape((90, 6)))
        ttt(np.arange(92 * 6).reshape((92, 6)))
        ttt(_to_str(np.arange(90 * 6).reshape((90, 6))))

        ttt(np.asarray(42))

        fff(np.arange(90 * 6).reshape((6, 90)))
        fff(np.arange(88 * 6).reshape((88, 6)))

        ttt(np.arange(6))
        ttt(np.arange(90))

    def test_node_and_arbitrary(self):
        ttt, fff = self.as_bool_asserts(lambda x: Shapes.NxA.matches(_dim, x))

        ttt(np.arange(6 * 111).reshape((6, 111)))
        ttt(np.arange(6 * 222).reshape((6, 222)))

        fff(np.arange(6 * 111).reshape((111, 6)))
        fff(np.arange(6 * 222).reshape((222, 6)))

        fff(np.arange(4 * 111).reshape((4, 111)))
        fff(np.arange(4 * 222).reshape((4, 222)))

    def test_arbitrary_and_node(self):
        ttt, fff = self.as_bool_asserts(lambda x: Shapes.AxN.matches(_dim, x))

        ttt(np.arange(6 * 111).reshape((111, 6)))
        ttt(np.arange(6 * 222).reshape((222, 6)))

        fff(np.arange(6 * 111).reshape((6, 111)))
        fff(np.arange(6 * 222).reshape((6, 222)))

        fff(np.arange(4 * 111).reshape((111, 4)))
        fff(np.arange(4 * 222).reshape((222, 4)))

    def adapt_test_framework(self, shape, cases):
        for i, (input_value, expected) in enumerate(cases):
            error = f"Failure in test case {i}: ({shape}, {input_value}, {expected})"
            if expected is None:
                # Using None to indicate that we expect the adaptation to fail;
                # and the failure mode is to raise ValueError.
                with self.assertRaises(ValueError, msg=error):
                    shape.adapt(_dim, input_value)
            else:
                # If expected is not None, anticipate adapation to be successful;
                # check returned value is a match.
                actual = shape.adapt(_dim, input_value)
                if self.assert_not_none(actual, msg=error):
                    np.testing.assert_array_equal(actual, expected, err_msg=error)

    def test_adapt_scalar(self):
        self.adapt_test_framework(
            Shapes.Scalar,
            [
                # Test S
                (np.asarray(42.0), np.asarray(42.0)),
                # Test higher dimensions
                (np.asarray([42.0, 43.0, 44.0]), None),
                (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), None),
            ],
        )

    def test_adapt_node(self):
        N = _dim.N
        node_values = np.asarray([41.0, 42.0, 43.0, 44.0, 45.0, 46.0])
        self.adapt_test_framework(
            Shapes.N,
            [
                # Test S
                (np.asarray(42.0), np.full(N, 42.0)),
                # Test N
                (node_values.copy(), node_values),
                # Test < N
                (np.arange(3), None),
                # Test > N
                (np.arange(30), None),
                # Test NxN
                (np.arange(N * N).reshape((N, N)), None),
                # Test higher dimension
                (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), None),
            ],
        )

    def test_adapt_time(self):
        T = _dim.T
        time_values = np.arange(T) + 40
        self.adapt_test_framework(
            Shapes.T,
            [
                # Test S
                (np.asarray(42.0), np.full(T, 42.0)),
                # Test T
                (time_values.copy(), time_values),
                # Test < T
                (np.arange(40) * 7, None),
                # Test > T
                (np.arange(200) * 13, np.arange(T) * 13),
                # Test TxT
                ((np.arange(T * T)).reshape((T, T)), None),
                # Test higher dimension
                (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), None),
            ],
        )

    def test_adapt_node_and_node(self):
        N = _dim.N
        nxn = (N, N)
        nxn_values = np.arange(N * N).reshape(nxn) + 42
        self.adapt_test_framework(
            Shapes.NxN,
            [
                # Test S
                (np.asarray(42.0), np.full((nxn), 42.0)),
                # Test < N (assume 4 to be less)
                (np.arange(4), None),
                # Test > N (assume 10 to be greater)
                (np.arange(10), None),
                # Test N
                (np.arange(N), None),
                # Test NxN
                (nxn_values.copy(), nxn_values),
                # Test Nx10 and 10xN
                (np.arange(N * 10).reshape((N, 10)), None),
                (np.arange(N * 10).reshape((10, N)), None),
                # Test higher dimension
                (np.arange(27).reshape((3, 3, 3)), None),
            ],
        )

    def test_adapt_time_and_node(self):
        T, N = _dim.T, _dim.N
        txn = (T, N)
        txt = (T, T)
        nxn = (N, N)
        txn_values = np.arange(T * N).reshape(txn) + 40
        self.adapt_test_framework(
            Shapes.TxN,
            [
                # Test S
                (np.asarray(42.0), np.full((txn), 42.0)),
                # Test < T and N (assume 4 to be less than either)
                (np.arange(4), None),
                # T < Test < N (assume 32 to be between them)
                (np.arange(32), None),
                # Test > T (assume 999 to be greater than either)
                (np.arange(999), np.tile(np.arange(T), (N, 1)).T),
                # Test N
                (np.arange(N), np.tile(np.arange(N), (T, 1))),
                # Test T
                (np.arange(T), np.tile(np.arange(T), (N, 1)).T),
                # Test NxN
                ((np.arange(N * N)).reshape(nxn), None),
                # Test TxT
                ((np.arange(T * T)).reshape(txt), None),
                # Test TxN
                (txn_values.copy(), txn_values),
                # Test higher dimension
                (np.arange(27).reshape((3, 3, 3)), None),
            ],
        )

    def test_adapt_node_and_arbitrary(self):
        arr1 = np.arange(6 * 111).reshape((6, 111))
        arr2 = np.arange(6 * 222).reshape((6, 222))
        arr3 = np.arange(6 * 333).reshape((6, 333))

        arr4 = np.arange(5 * 111).reshape((5, 111))
        arr5 = np.arange(111)
        arr6 = np.arange(6)

        self.adapt_test_framework(
            Shapes.NxA,
            [
                (arr1, arr1),
                (arr2, arr2),
                (arr3, arr3),
                (arr4, None),
                (arr5, None),
                (arr6, None),
            ],
        )

    def test_adapt_arbitrary_and_node(self):
        arr1 = np.arange(6 * 111).reshape((111, 6))
        arr2 = np.arange(6 * 222).reshape((222, 6))
        arr3 = np.arange(6 * 333).reshape((333, 6))

        arr4 = np.arange(5 * 111).reshape((111, 5))
        arr5 = np.arange(111)
        arr6 = np.arange(6)

        self.adapt_test_framework(
            Shapes.AxN,
            [
                (arr1, arr1),
                (arr2, arr2),
                (arr3, arr3),
                (arr4, None),
                (arr5, None),
                (arr6, None),
            ],
        )


class TestParseShape(unittest.TestCase):
    def test_successful(self):
        eq = self.assertEqual
        eq(parse_shape("Scalar"), Shapes.Scalar)
        eq(parse_shape("T"), Shapes.T)
        eq(parse_shape("N"), Shapes.N)
        eq(parse_shape("NxN"), Shapes.NxN)
        eq(parse_shape("TxN"), Shapes.TxN)
        eq(parse_shape("AxN"), Shapes.AxN)
        eq(parse_shape("NxA"), Shapes.NxA)

    def test_failure(self):
        def test(s):
            with self.assertRaises(ValueError):
                parse_shape(s)

        test("A")
        test("TxA")
        test("NxNx32")
        test("TxNxN")
        test("TxNxNx4")
        test("A")
        test("3BC")
        test("Tx3N")
        test("3T")
        test("T3")
        test("N3T")
        test("TxT")
        test("NxN3")
        test("3TxN")
        test("TxN3T")
        test("Tx3T")
        test("NTxN")
        test("NxTxN")
