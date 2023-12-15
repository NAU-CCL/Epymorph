# pylint: disable=missing-docstring
import unittest
from functools import reduce
from typing import Any, TypeGuard, TypeVar

import numpy as np

from epymorph.data_shape import Shapes, SimDimensions, parse_shape

_to_str = np.vectorize(str)

_dim = SimDimensions.build(
    tau_step_lengths=[0.5, 0.5],
    days=90,
    nodes=6,
    compartments=3,
    events=2
)

T = TypeVar('T')


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
        ttt, fff = self.as_bool_asserts(
            lambda x: Shapes.S.matches(_dim, x, False)
        )

        ttt(np.asarray(1))
        ttt(np.asarray(3.14159))
        ttt(np.asarray("this is a string"))

        fff(np.array([1]))
        fff(np.array([1, 2, 3]))
        fff(np.arange(9).reshape((3, 3)))

    def test_time(self):
        ttt, fff = self.as_bool_asserts(
            lambda x, bc=False: Shapes.T.matches(_dim, x, bc)
        )

        ttt(np.arange(90), False)
        ttt(np.arange(90), True)
        ttt(np.arange(99), False)
        ttt(np.arange(99), True)
        ttt(_to_str(np.arange(90)))

        ttt(np.asarray(42), True)
        fff(np.asarray(42), False)

        fff(np.arange(6), False)
        fff(np.arange(6), True)
        fff(np.arange(90 * 2).reshape((90, 2)))

    def test_node(self):
        ttt, fff = self.as_bool_asserts(
            lambda x, bc=False: Shapes.N.matches(_dim, x, bc)
        )

        ttt(np.arange(6))
        ttt(_to_str(np.arange(6)))

        ttt(np.asarray(42), True)
        fff(np.asarray(42), False)

        fff(np.arange(5))
        fff(np.arange(7))
        fff(np.arange(90))
        fff(np.arange(6 * 6 * 6).reshape((6, 6, 6)))

    def test_time_and_node(self):
        ttt, fff = self.as_bool_asserts(
            lambda x, bc=False: Shapes.TxN.matches(_dim, x, bc)
        )

        ttt(np.arange(90 * 6).reshape((90, 6)))
        ttt(np.arange(92 * 6).reshape((92, 6)))
        ttt(_to_str(np.arange(90 * 6).reshape((90, 6))))

        ttt(np.asarray(42), True)
        fff(np.asarray(42), False)

        fff(np.arange(90 * 6).reshape((6, 90)))
        fff(np.arange(88 * 6).reshape((88, 6)))

        ttt(np.arange(6), True)
        fff(np.arange(6), False)
        ttt(np.arange(90), True)
        fff(np.arange(90), False)

    def test_arbitrary(self):
        self.assertTrue(
            Shapes.A(0).matches(_dim, np.array([1, 2, 3]), False)
        )
        self.assertTrue(
            Shapes.A(1).matches(_dim, np.array([1, 2, 3]), False)
        )
        self.assertTrue(
            Shapes.A(2).matches(_dim, np.array([1, 2, 3]), False)
        )
        self.assertFalse(
            Shapes.A(3).matches(_dim, np.asarray(1), False)
        )
        self.assertFalse(
            Shapes.A(3).matches(_dim, np.array([1, 2, 3]), False)
        )
        self.assertFalse(
            Shapes.A(1).matches(_dim, np.arange(9).reshape((3, 3)), False)
        )

    def test_time_and_arbitrary(self):
        self.assertTrue(
            Shapes.TxA(2).matches(_dim, np.arange(90 * 3).reshape((90, 3)), False)
        )
        self.assertTrue(
            Shapes.TxA(2).matches(_dim, np.arange(90 * 6).reshape((90, 6)), False)
        )
        self.assertFalse(
            Shapes.TxA(2).matches(_dim, np.arange(90 * 2).reshape((90, 2)), False)
        )
        self.assertFalse(
            Shapes.TxA(2).matches(_dim, np.arange(90), False)
        )
        self.assertFalse(
            Shapes.TxA(2).matches(_dim, np.arange(
                90 * 3 * 3).reshape((90, 3, 3)), False)
        )

    def test_node_and_arbitrary(self):
        self.assertTrue(
            Shapes.NxA(2).matches(_dim, np.arange(6 * 3).reshape((6, 3)), False)
        )
        self.assertTrue(
            Shapes.NxA(2).matches(_dim, np.arange(6 * 6).reshape((6, 6)), False)
        )
        self.assertFalse(
            Shapes.NxA(2).matches(_dim, np.arange(6 * 2).reshape((6, 2)), False)
        )
        self.assertFalse(
            Shapes.NxA(2).matches(_dim, np.arange(6), False)
        )
        self.assertFalse(
            Shapes.NxA(2).matches(_dim, np.arange(6 * 3 * 3).reshape((6, 3, 3)), False)
        )

    def test_time_and_node_and_arbitrary(self):
        self.assertTrue(
            Shapes.TxNxA(0).matches(
                _dim, np.arange(90 * 6 * 2).reshape((90, 6, 2)), False
            )
        )
        self.assertFalse(
            Shapes.TxNxA(4).matches(
                _dim, np.arange(90 * 6 * 2).reshape((90, 6, 2)), False
            )
        )

    def adapt_test_framework(self, shape, cases):
        for i, (input_value, broadcast, expected) in enumerate(cases):
            error = f"Failure in test case {i}: ({shape}, {input_value}, {broadcast}, {expected})"
            actual = shape.adapt(_dim, input_value, broadcast)
            if expected is None:
                self.assertIsNone(actual, error)
            elif self.assert_not_none(actual, error):
                np.testing.assert_array_equal(actual, expected, error)

    def test_adapt_scalar(self):
        self.adapt_test_framework(Shapes.S, [
            # Test S
            (np.asarray(42.0), True, np.asarray(42.0)),
            (np.asarray(42.0), False, np.asarray(42.0)),
            # Test higher dimensions
            (np.asarray([42.0, 43.0, 44.0]), True, None),
            (np.asarray([42.0, 43.0, 44.0]), False, None),
            (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), True, None),
            (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), False, None),
        ])

    def test_adapt_node(self):
        node_values = np.asarray([41.0, 42.0, 43.0, 44.0, 45.0, 46.0])
        self.adapt_test_framework(Shapes.N, [
            # Test S
            (np.asarray(42.0), True, np.full(_dim.nodes, 42.0)),
            (np.asarray(42.0), False, None),
            # Test N
            (node_values.copy(), True, node_values),
            (node_values.copy(), False, node_values),
            # Test < N
            (np.arange(3), True, None),
            (np.arange(3), False, None),
            # Test > N
            (np.arange(30), True, None),
            (np.arange(30), False, None),
            # Test NxN
            (np.arange(_dim.nodes * _dim.nodes).reshape((_dim.nodes, _dim.nodes)), True, None),
            (np.arange(_dim.nodes * _dim.nodes).reshape((_dim.nodes, _dim.nodes)), False, None),
            # Test higher dimension
            (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), True, None),
            (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), False, None),
        ])

    def test_adapt_time(self):
        time_values = np.arange(_dim.days) + 40
        self.adapt_test_framework(Shapes.T, [
            # Test S
            (np.asarray(42.0), True, np.full(_dim.days, 42.0)),
            (np.asarray(42.0), False, None),
            # Test T
            (time_values.copy(), True, time_values),
            (time_values.copy(), False, time_values),
            # Test < T
            (np.arange(40) * 7, True, None),
            (np.arange(40) * 7, False, None),
            # Test > T
            (np.arange(200) * 13, True, np.arange(_dim.days) * 13),
            (np.arange(200) * 13, False, np.arange(_dim.days) * 13),
            # Test TxT
            ((np.arange(_dim.days * _dim.days)).reshape((_dim.days, _dim.days)), True, None),
            ((np.arange(_dim.days * _dim.days)).reshape((_dim.days, _dim.days)), False, None),
            # Test higher dimension
            (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), True, None),
            (np.asarray([[42.0, 43.0, 44.0], [1.0, 2.0, 3.0]]), False, None),
        ])

    def test_adapt_node_and_node(self):
        nxn = (_dim.nodes, _dim.nodes)
        nxn_values = np.arange(_dim.nodes * _dim.nodes).reshape(nxn) + 42
        self.adapt_test_framework(Shapes.NxN, [
            # Test S
            (np.asarray(42.0), True, np.full((nxn), 42.0)),
            (np.asarray(42.0), False, None),
            # Test < N (assume 4 to be less)
            (np.arange(4), True, None),
            (np.arange(4), False, None),
            # Test > N (assume 10 to be greater)
            (np.arange(10), True, None),
            (np.arange(10), False, None),
            # Test N
            (np.arange(_dim.nodes), True, None),
            (np.arange(_dim.nodes), False, None),
            # Test NxN
            (nxn_values.copy(), True, nxn_values),
            (nxn_values.copy(), False, nxn_values),
            # Test Nx10 and 10xN
            (np.arange(_dim.nodes * 10).reshape((_dim.nodes, 10)), True, None),
            (np.arange(_dim.nodes * 10).reshape((_dim.nodes, 10)), False, None),
            (np.arange(_dim.nodes * 10).reshape((10, _dim.nodes)), True, None),
            (np.arange(_dim.nodes * 10).reshape((10, _dim.nodes)), False, None),
            # Test higher dimension
            (np.arange(27).reshape((3, 3, 3)), True, None),
            (np.arange(27).reshape((3, 3, 3)), False, None),
        ])

    def test_adapt_time_and_node(self):
        txn = (_dim.days, _dim.nodes)
        txt = (_dim.days, _dim.days)
        nxn = (_dim.nodes, _dim.nodes)
        txn_values = np.arange(_dim.days * _dim.nodes).reshape(txn) + 40
        self.adapt_test_framework(Shapes.TxN, [
            # Test S
            (np.asarray(42.0), True, np.full((txn), 42.0)),
            (np.asarray(42.0), False, None),
            # Test < T and N (assume 4 to be less than either)
            (np.arange(4), True, None),
            (np.arange(4), False, None),
            # T < Test < N (assume 32 to be between them)
            (np.arange(32), True, None),
            (np.arange(32), False, None),
            # Test > T (assume 999 to be greater than either)
            (np.arange(999), True, np.tile(np.arange(_dim.days), (_dim.nodes, 1)).T),
            (np.arange(999), False, None),
            # Test N
            (np.arange(_dim.nodes), True, np.tile(np.arange(_dim.nodes), (_dim.days, 1))),
            (np.arange(_dim.nodes), False, None),
            # Test T
            (np.arange(_dim.days), True, np.tile(np.arange(_dim.days), (_dim.nodes, 1)).T),
            (np.arange(_dim.days), False, None),
            # Test NxN
            ((np.arange(_dim.nodes * _dim.nodes)).reshape(nxn), True, None),
            ((np.arange(_dim.nodes * _dim.nodes)).reshape(nxn), False, None),
            # Test TxT
            ((np.arange(_dim.days * _dim.days)).reshape(txt), True, None),
            ((np.arange(_dim.days * _dim.days)).reshape(txt), False, None),
            # Test TxN
            (txn_values.copy(), True, txn_values),
            (txn_values.copy(), False, txn_values),
            # Test higher dimension
            (np.arange(27).reshape((3, 3, 3)), True, None),
            (np.arange(27).reshape((3, 3, 3)), False, None),
        ])

    def test_adapt_abitrary(self):
        # NOTE: A(2) is expecting to be able to access index 2 (as in `values[2]`)
        # which implies our data must have at least 3 items.
        self.adapt_test_framework(Shapes.A(2), [
            # Test S
            (np.asarray(42.0), True, None),
            (np.asarray(42.0), False, None),
            # Test 3
            (np.array([42.0, 43.0, 44.0]), True, np.array([42.0, 43.0, 44.0])),
            (np.array([42.0, 43.0, 44.0]), False, np.array([42.0, 43.0, 44.0])),
            # Test 4
            (np.array([42.0, 43.0, 44.0, 45.0]), True,
             np.array([42.0, 43.0, 44.0, 45.0])),
            (np.array([42.0, 43.0, 44.0, 45.0]), False,
             np.array([42.0, 43.0, 44.0, 45.0])),
            # Test higher dimensions
            (np.arange(9).reshape((3, 3)), True, None),
            (np.arange(9).reshape((3, 3)), False, None),
            (np.arange(27).reshape((3, 3, 3)), True, None),
            (np.arange(27).reshape((3, 3, 3)), False, None),
        ])

    def test_adapt_node_and_abitrary(self):
        self.adapt_test_framework(Shapes.NxA(2), [
            # Test S
            (np.asarray(42.0), True, None),
            (np.asarray(42.0), False, None),
            # Test 3
            (np.array([42.0, 43.0, 44.0]), True, np.tile(
                np.array([42.0, 43.0, 44.0]), (_dim.nodes, 1))),
            (np.array([42.0, 43.0, 44.0]), False, None),
            # Test 4
            (np.array([42.0, 43.0, 44.0, 45.0]), True, np.tile(
                np.array([42.0, 43.0, 44.0, 45.0]), (_dim.nodes, 1))),
            (np.array([42.0, 43.0, 44.0, 45.0]), False, None),
            # Test other dimensions
            (np.arange(9).reshape((3, 3)), True, None),
            (np.arange(9).reshape((3, 3)), False, None),
            (np.arange(27).reshape((3, 3, 3)), True, None),
            (np.arange(27).reshape((3, 3, 3)), False, None),
        ])

    def test_adapt_time_and_abitrary(self):
        tx3_values = np.arange(3 * _dim.days).reshape((_dim.days, 3))
        self.adapt_test_framework(Shapes.TxA(2), [
            # Test S
            (np.asarray(42.0), True, None),
            (np.asarray(42.0), False, None),
            # Test 2
            (np.arange(2), True, None),
            (np.arange(2), False, None),
            # Test 3
            (np.arange(3), True, np.tile(np.arange(3), (_dim.days, 1))),
            (np.arange(3), False, None),
            # Test 4
            (np.arange(4), True, np.tile(np.arange(4), (_dim.days, 1))),
            (np.arange(4), False, None),
            # Test Tx3
            (tx3_values, True, tx3_values),
            (tx3_values, False, tx3_values),
            # Test (>T)x3
            (np.arange(300 * 3).reshape((300, 3)), True, tx3_values),
            (np.arange(300 * 3).reshape((300, 3)), False, tx3_values),
            # Test other dimensions
            (np.arange(9).reshape((3, 3)), True, None),
            (np.arange(9).reshape((3, 3)), False, None),
            (np.arange(27).reshape((3, 3, 3)), True, None),
            (np.arange(27).reshape((3, 3, 3)), False, None),
        ])

    def test_adapt_time_and_node_and_abitrary(self):
        txnx3_values = shaped_arange((_dim.days, _dim.nodes, 3))
        self.adapt_test_framework(Shapes.TxNxA(2), [
            # Test S
            (np.asarray(42.0), True, None),
            (np.asarray(42.0), False, None),
            # Test 2
            (np.arange(2), True, None),
            (np.arange(2), False, None),
            # Test 3
            (np.arange(3), True, np.tile(np.arange(3), (_dim.days, _dim.nodes, 1))),
            (np.arange(3), False, None),
            # Test 4
            (np.arange(4), True, np.tile(np.arange(4), (_dim.days, _dim.nodes, 1))),
            (np.arange(4), False, None),
            # Test Tx3
            (txnx3_values, True, txnx3_values),
            (txnx3_values, False, txnx3_values),
            # Test (>T)x3
            (shaped_arange((300, 3)), True, np.tile(
                shaped_arange((_dim.days, 1, 3)), (1, _dim.nodes, 1))),
            (shaped_arange((300, 3)), False, None),
            # Test Nx3
            (shaped_arange((_dim.nodes, 3)), True, np.tile(
                shaped_arange((_dim.nodes, 3)), (_dim.days, 1, 1))),
            (shaped_arange((_dim.nodes, 3)), False, None),
            # Test (>N)x3
            (shaped_arange((10, 3)), True, None),
            (shaped_arange((10, 3)), False, None),
            # Test (>T)xNx3
            (shaped_arange((300, _dim.nodes, 3)), True, txnx3_values),
            (shaped_arange((300, _dim.nodes, 3)), False, txnx3_values),
            # Test Tx(>N)x3
            (shaped_arange((_dim.days, 10, 3)), True, None),
            (shaped_arange((_dim.days, 10, 3)), False, None),
            # Test other dimensions
            (shaped_arange((3, 3, 3, 3)), True, None),
            (shaped_arange((3, 3, 3, 3)), False, None),
        ])


class TestParseShape(unittest.TestCase):
    def test_successful(self):
        eq = self.assertEqual
        eq(parse_shape('S'), Shapes.S)
        eq(parse_shape('0'), Shapes.A(0))
        eq(parse_shape('2'), Shapes.A(2))
        eq(parse_shape('3'), Shapes.A(3))
        eq(parse_shape('13'), Shapes.A(13))
        eq(parse_shape('T'), Shapes.T)
        eq(parse_shape('Tx9'), Shapes.TxA(9))
        eq(parse_shape('N'), Shapes.N)
        eq(parse_shape('Nx20'), Shapes.NxA(20))
        eq(parse_shape('TxN'), Shapes.TxN)
        eq(parse_shape('TxNx2'), Shapes.TxNxA(2))
        eq(parse_shape('Tx0'), Shapes.TxA(0))
        eq(parse_shape('Tx1'), Shapes.TxA(1))
        eq(parse_shape('Nx1'), Shapes.NxA(1))
        eq(parse_shape('TxNx1'), Shapes.TxNxA(1))

    def test_failure(self):
        def test(s):
            with self.assertRaises(ValueError):
                parse_shape(s)

        test('NxN')
        test('NxNx32')
        test('TxNxN')
        test('TxNxNx4')
        test('A')
        test('3BC')
        test('Tx3N')
        test('3T')
        test('T3')
        test('N3T')
        test('TxT')
        test('NxN3')
        test('3TxN')
        test('TxN3T')
        test('Tx3T')
        test('NTxN')
        test('NxTxN')
