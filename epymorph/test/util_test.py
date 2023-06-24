import unittest

import numpy as np
from dateutil.relativedelta import relativedelta

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

    def test_parse_duration(self):
        act1 = util.parse_duration("30d")
        exp1 = relativedelta(days=30)
        self.assertEqual(act1, exp1)

        act2 = util.parse_duration("3w")
        exp2 = relativedelta(weeks=3)
        self.assertEqual(act2, exp2)

        act3 = util.parse_duration("7m")
        exp3 = relativedelta(months=7)
        self.assertEqual(act3, exp3)

        act4 = util.parse_duration("999y")
        exp4 = relativedelta(years=999)
        self.assertEqual(act4, exp4)

    def test_filter_unique(self):
        act = util.filter_unique(['a', 'b', 'b', 'c', 'a'])
        exp = ['a', 'b', 'c']
        self.assertListEqual(act, exp)

    def test_list_not_none(self):
        act = util.list_not_none(['a', None, 'b', None, None, 'c', None])
        exp = ['a', 'b', 'c']
        self.assertListEqual(act, exp)
