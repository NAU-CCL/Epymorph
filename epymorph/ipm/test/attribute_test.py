import unittest

from epymorph.ipm.attribute import validate_shape


class TestAttributeShape(unittest.TestCase):
    def test_successful(self):
        def test(s):
            validate_shape(s)

        test('S')
        test('0')
        test('2')
        test('3')
        test('13')
        test('T')
        test('Tx9')
        test('N')
        test('Nx20')
        test('TxN')
        test('TxNx2')
        test('Tx0')
        test('Tx1')
        test('Nx1')
        test('TxNx1')

    def test_failure(self):
        def test(s):
            with self.assertRaises(ValueError):
                validate_shape(s)

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
