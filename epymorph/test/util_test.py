import unittest

import numpy as np

import epymorph.util as util


class TestUtil(unittest.TestCase):
    def test_identity(self):
        tests = [1, "hey", [1, 2, 3], {"foo": "bar"}]
        for t in tests:
            self.assertEqual(t, util.identity(t))

    def test_stutter(self):
        act = list(util.stutter(['a', 'b', 'c'], 3))
        exp = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
        self.assertEqual(act, exp)

    def test_stridesum(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.int_)

        act1 = util.stridesum(arr, 2)
        exp1 = np.array([3, 7, 11, 15, 9])
        self.assertTrue(all(np.equal(act1, exp1)))

        act2 = util.stridesum(arr, 5)
        exp2 = np.array([15, 30])
        self.assertTrue(all(np.equal(act2, exp2)))
