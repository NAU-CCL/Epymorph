# pylint: disable=missing-docstring
import unittest

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


class DataShape(unittest.TestCase):
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
