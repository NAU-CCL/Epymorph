import unittest

import numpy as np

from epymorph.context import Shapes, SimDimension, parse_shape


class TestDims(SimDimension):
    def __init__(self, dims: tuple[int, int, int, int, int]):
        D, T, N, C, E = dims
        self.nodes = N
        self.compartments = C
        self.events = E
        self.ticks = T
        self.days = D
        self.TNCE = T, N, C, E


DIM = TestDims((90, 180, 6, 3, 2))


class DataShape(unittest.TestCase):
    def test_scalar(self):
        S = Shapes.S
        self.assertTrue(S.matches(DIM, 1))
        self.assertTrue(S.matches(DIM, 3.14159))
        self.assertTrue(S.matches(DIM, "this is a string"))
        self.assertTrue(S.matches(DIM, np.int64(42)))

        self.assertFalse(S.matches(DIM, np.array([1])))
        self.assertFalse(S.matches(DIM, np.array([1, 2, 3])))
        self.assertFalse(S.matches(DIM, [1, 2, 3]))

    def test_time(self):
        T = Shapes.T
        self.assertTrue(T.matches(DIM, np.arange(90)))
        self.assertTrue(T.matches(DIM, np.arange(99)))
        to_str = np.vectorize(str)
        self.assertTrue(T.matches(DIM, to_str(np.arange(90))))

        self.assertFalse(T.matches(DIM, 101))
        self.assertFalse(T.matches(DIM, np.arange(6)))
        self.assertFalse(T.matches(DIM, np.arange(90 * 2).reshape((90, 2))))

    def test_node(self):
        N = Shapes.N
        self.assertTrue(N.matches(DIM, np.arange(6)))
        to_str = np.vectorize(str)
        self.assertTrue(N.matches(DIM, to_str(np.arange(6))))

        self.assertFalse(N.matches(DIM, 101))
        self.assertFalse(N.matches(DIM, np.arange(5)))
        self.assertFalse(N.matches(DIM, np.arange(7)))
        self.assertFalse(N.matches(DIM, np.arange(90)))
        self.assertFalse(N.matches(DIM, np.arange(6 * 6 * 6).reshape((6, 6, 6))))

    def test_time_and_node(self):
        TxN = Shapes.TxN
        self.assertTrue(TxN.matches(DIM, np.arange(90 * 6).reshape((90, 6))))
        self.assertTrue(TxN.matches(DIM, np.arange(92 * 6).reshape((92, 6))))
        to_str = np.vectorize(str)
        self.assertTrue(TxN.matches(DIM, to_str(np.arange(90 * 6).reshape((90, 6)))))

        self.assertFalse(TxN.matches(DIM, 101))
        self.assertFalse(TxN.matches(DIM, np.arange(90 * 6).reshape((6, 90))))
        self.assertFalse(TxN.matches(DIM, np.arange(88 * 6).reshape((88, 6))))
        self.assertFalse(TxN.matches(DIM, np.arange(6)))
        self.assertFalse(TxN.matches(DIM, np.arange(90)))

    def test_arbitrary(self):
        A = Shapes.A
        self.assertTrue(A[0].matches(DIM, np.array([1, 2, 3])))
        self.assertTrue(A[1].matches(DIM, np.array([1, 2, 3])))
        self.assertTrue(A[2].matches(DIM, np.array([1, 2, 3])))
        self.assertFalse(A[3].matches(DIM, np.array([1, 2, 3])))
        self.assertFalse(A[1].matches(DIM, np.arange(9).reshape((3, 3))))

        self.assertTrue(Shapes.T[2].matches(DIM, np.arange(90 * 3).reshape((90, 3))))
        self.assertTrue(Shapes.T[2].matches(DIM, np.arange(90 * 6).reshape((90, 6))))
        self.assertFalse(Shapes.T[2].matches(DIM, np.arange(90 * 2).reshape((90, 2))))
        self.assertFalse(Shapes.T[2].matches(DIM, np.arange(90)))
        self.assertFalse(
            Shapes.T[2].matches(DIM, np.arange(90 * 3 * 3).reshape((90, 3, 3))))

        self.assertTrue(Shapes.N[2].matches(DIM, np.arange(6 * 3).reshape((6, 3))))
        self.assertTrue(Shapes.N[2].matches(DIM, np.arange(6 * 6).reshape((6, 6))))
        self.assertFalse(Shapes.N[2].matches(DIM, np.arange(6 * 2).reshape((6, 2))))
        self.assertFalse(Shapes.N[2].matches(DIM, np.arange(6)))
        self.assertFalse(
            Shapes.N[2].matches(DIM, np.arange(6 * 3 * 3).reshape((6, 3, 3))))

        self.assertTrue(
            Shapes.TxN[0].matches(DIM, np.arange(90 * 6 * 2).reshape((90, 6, 2))))
        self.assertFalse(
            Shapes.TxN[4].matches(DIM, np.arange(90 * 6 * 2).reshape((90, 6, 2))))


class TestParseShape(unittest.TestCase):
    def test_successful(self):
        eq = self.assertEqual
        eq(parse_shape('S'), Shapes.S)
        eq(parse_shape('0'), Shapes.A[0])
        eq(parse_shape('2'), Shapes.A[2])
        eq(parse_shape('3'), Shapes.A[3])
        eq(parse_shape('13'), Shapes.A[13])
        eq(parse_shape('T'), Shapes.T)
        eq(parse_shape('Tx9'), Shapes.T[9])
        eq(parse_shape('N'), Shapes.N)
        eq(parse_shape('Nx20'), Shapes.N[20])
        eq(parse_shape('TxN'), Shapes.TxN)
        eq(parse_shape('TxNx2'), Shapes.TxN[2])
        eq(parse_shape('Tx0'), Shapes.T[0])
        eq(parse_shape('Tx1'), Shapes.T[1])
        eq(parse_shape('Nx1'), Shapes.N[1])
        eq(parse_shape('TxNx1'), Shapes.TxN[1])

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
