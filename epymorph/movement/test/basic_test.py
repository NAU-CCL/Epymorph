import unittest

import numpy as np

from epymorph.movement.basic import HOME_TICK, BasicLocation, Cohort


class TestBasicLocation(unittest.TestCase):
    def test_normalize_01(self):
        act = BasicLocation(
            index=0,
            cohorts=[
                Cohort(np.array([100]), 0, HOME_TICK),
                Cohort(np.array([200]), 0, HOME_TICK),
            ])

        exp = BasicLocation(
            index=0,
            cohorts=[
                Cohort(np.array([300]), 0, HOME_TICK),
            ])

        act.normalize()
        self.assertEqual(act, exp)

    def test_normalize_02(self):
        act = BasicLocation(
            index=0,
            cohorts=[
                Cohort(np.array([200]), 0, HOME_TICK),
                Cohort(np.array([75]), 2, 2),
                Cohort(np.array([100]), 0, HOME_TICK),
                Cohort(np.array([50]), 1, 2),
            ])

        exp = BasicLocation(
            index=0,
            cohorts=[
                Cohort(np.array([300]), 0, HOME_TICK),
                Cohort(np.array([50]), 1, 2),
                Cohort(np.array([75]), 2, 2),
            ])

        act.normalize()
        self.assertEqual(act, exp)

    def test_normalize_03(self):
        act = BasicLocation(
            index=0,
            cohorts=[
                Cohort(np.array([100]), 0, HOME_TICK),
                Cohort(np.array([100]), 0, HOME_TICK),
                Cohort(np.array([100]), 0, HOME_TICK),
                Cohort(np.array([100]), 0, HOME_TICK),
                Cohort(np.array([100]), 0, HOME_TICK),
            ]
        )

        exp = BasicLocation(
            index=0,
            cohorts=[
                Cohort(np.array([500]), 0, HOME_TICK),
            ])

        act.normalize()
        self.assertEqual(exp, act)
