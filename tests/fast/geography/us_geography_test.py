# ruff: noqa: PT009,PT027
import unittest

import numpy as np

from epymorph.error import GeographyError
from epymorph.geography.us_geography import (
    BLOCK,
    BLOCK_GROUP,
    CENSUS_GRANULARITY,
    COUNTY,
    STATE,
    TRACT,
)


class TestCensusGranularity(unittest.TestCase):
    def test_is_nested(self):
        # a triangular array perfectly describes the pattern of truth
        expected = np.tri(5)
        for i, test in enumerate(CENSUS_GRANULARITY):
            for j, outer in enumerate(CENSUS_GRANULARITY):
                if expected[i, j] == 1:
                    self.assertTrue(test.is_nested(outer.name))
                else:
                    self.assertFalse(test.is_nested(outer.name))

    def test_matches(self):
        self.assertTrue(STATE.matches("04"))
        self.assertTrue(COUNTY.matches("04003"))
        self.assertTrue(TRACT.matches("04003999999"))
        self.assertTrue(BLOCK_GROUP.matches("040039999998"))
        self.assertTrue(BLOCK.matches("040039999998777"))

        self.assertFalse(STATE.matches("0"))
        self.assertFalse(STATE.matches(""))
        self.assertFalse(STATE.matches("04003"))
        self.assertFalse(STATE.matches("AZ"))
        self.assertFalse(COUNTY.matches("04"))
        self.assertFalse(COUNTY.matches("04003999999"))

    def test_extract(self):
        self.assertEqual("04", STATE.extract("04"))
        self.assertEqual("04", STATE.extract("04003"))
        self.assertEqual("04", STATE.extract("04003999999"))
        self.assertEqual("04", STATE.extract("040039999998"))
        self.assertEqual("04", STATE.extract("040039999998777"))

        self.assertEqual("003", COUNTY.extract("04003"))
        self.assertEqual("003", COUNTY.extract("04003999999"))
        self.assertEqual("003", COUNTY.extract("040039999998"))
        self.assertEqual("003", COUNTY.extract("040039999998777"))
        with self.assertRaises(GeographyError):
            COUNTY.extract("04")

        self.assertEqual("999999", TRACT.extract("04003999999"))
        self.assertEqual("999999", TRACT.extract("040039999998"))
        self.assertEqual("999999", TRACT.extract("040039999998777"))
        with self.assertRaises(GeographyError):
            TRACT.extract("04")

        self.assertEqual("8", BLOCK_GROUP.extract("040039999998"))
        self.assertEqual("8", BLOCK_GROUP.extract("040039999998777"))
        with self.assertRaises(GeographyError):
            BLOCK_GROUP.extract("04")

        self.assertEqual("8777", BLOCK.extract("040039999998777"))
        with self.assertRaises(GeographyError):
            BLOCK.extract("04")

    def test_truncate(self):
        self.assertEqual("04", STATE.truncate("04"))
        self.assertEqual("04", STATE.truncate("04003"))
        self.assertEqual("04", STATE.truncate("04003999999"))
        self.assertEqual("04", STATE.truncate("040039999998"))
        self.assertEqual("04", STATE.truncate("040039999998777"))

        self.assertEqual("04003", COUNTY.truncate("04003"))
        self.assertEqual("04003", COUNTY.truncate("04003999999"))
        self.assertEqual("04003", COUNTY.truncate("040039999998"))
        self.assertEqual("04003", COUNTY.truncate("040039999998777"))

        self.assertEqual("04003999999", TRACT.truncate("04003999999"))
        self.assertEqual("04003999999", TRACT.truncate("040039999998"))
        self.assertEqual("04003999999", TRACT.truncate("040039999998777"))

        self.assertEqual("040039999998", BLOCK_GROUP.truncate("040039999998"))
        self.assertEqual("040039999998", BLOCK_GROUP.truncate("040039999998777"))

        self.assertEqual("040039999998777", BLOCK.truncate("040039999998777"))

    def test_truncate_unique(self):
        exp = ["08", "35", "04"]
        act = list(
            STATE.truncate_unique(["08001", "35", "04003", "08002", "04005", "35005"])
        )
        self.assertEqual(exp, act)

    def test_decompose(self):
        self.assertEqual(("04",), STATE.decompose("04"))
        self.assertEqual(("04", "003"), COUNTY.decompose("04003"))
        self.assertEqual(("04", "003", "999999"), TRACT.decompose("04003999999"))
        self.assertEqual(
            ("04", "003", "999999", "8"), BLOCK_GROUP.decompose("040039999998")
        )
        self.assertEqual(
            ("04", "003", "999999", "8", "8777"), BLOCK.decompose("040039999998777")
        )

        with self.assertRaises(GeographyError):
            STATE.decompose("04013")

        with self.assertRaises(GeographyError):
            TRACT.decompose("04013")

    def test_grouped(self):
        expected = {
            "04004": np.array(["04004111111", "04004222222", "04004333333"]),
            "04013": np.array(["04013444444", "04013555555", "04013666666"]),
        }
        actual = COUNTY.grouped(
            np.array(
                [
                    "04004111111",
                    "04004222222",
                    "04004333333",
                    "04013444444",
                    "04013555555",
                    "04013666666",
                ]
            )
        )
        self.assertSetEqual(set(expected.keys()), set(actual.keys()))
        np.testing.assert_array_equal(expected["04004"], actual["04004"])
        np.testing.assert_array_equal(expected["04013"], actual["04013"])
