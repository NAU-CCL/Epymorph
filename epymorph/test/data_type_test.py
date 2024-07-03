# pylint: disable=missing-docstring
import unittest

import numpy as np

from epymorph.data_type import dtype_as_np, dtype_check, dtype_str


class DataTypeTest(unittest.TestCase):

    def test_dtype_as_np(self):
        self.assertEqual(dtype_as_np(int), np.int64)
        self.assertEqual(dtype_as_np(float), np.float64)
        self.assertEqual(dtype_as_np(str), np.str_)

        struct = [('foo', float), ('bar', int), ('baz', str)]
        self.assertEqual(dtype_as_np(struct), np.dtype(struct))

    def test_dtype_str(self):
        self.assertEqual(dtype_str(int), "int")
        self.assertEqual(dtype_str(float), "float")
        self.assertEqual(dtype_str(str), "str")

        struct = [('foo', float), ('bar', int), ('baz', str)]
        self.assertEqual(dtype_str(struct), "[(foo, float), (bar, int), (baz, str)]")

    def test_dtype_check(self):
        self.assertTrue(dtype_check(int, 1))
        self.assertTrue(dtype_check(int, -32))
        self.assertTrue(dtype_check(float, -0.1))
        self.assertTrue(dtype_check(float, 191827312.231234))
        self.assertTrue(dtype_check(str, "hi"))
        self.assertTrue(dtype_check(str, ""))
        self.assertTrue(dtype_check([('x', int), ('y', int)], (1, 2)))
        self.assertTrue(dtype_check([('a', str), ('b', float)], ("hi", 9273.3)))

        self.assertFalse(dtype_check(int, "hi"))
        self.assertFalse(dtype_check(int, 42.42))
        self.assertFalse(dtype_check(int, (1, 2, 3)))

        self.assertFalse(dtype_check(float, "hi"))
        self.assertFalse(dtype_check(float, 1))
        self.assertFalse(dtype_check(float, 8273))
        self.assertFalse(dtype_check(float, (32.0, 12.7, 99.9)))

        dt1 = [('x', int), ('y', int)]
        self.assertFalse(dtype_check(dt1, 1))
        self.assertFalse(dtype_check(dt1, 78923.1))
        self.assertFalse(dtype_check(dt1, "hi"))
        self.assertFalse(dtype_check(dt1, ()))
        self.assertFalse(dtype_check(dt1, (1, 237.8)))
        self.assertFalse(dtype_check(dt1, (1, 2, 3)))

        dt2 = (('x', int), ('y', int))
        self.assertFalse(dtype_check(dt2, 1))
        self.assertFalse(dtype_check(dt2, 78923.1))
        self.assertFalse(dtype_check(dt2, "hi"))
        self.assertFalse(dtype_check(dt2, ()))
        self.assertFalse(dtype_check(dt2, (1, 237.8)))
        self.assertFalse(dtype_check(dt2, (1, 2, 3)))
