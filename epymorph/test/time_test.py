import unittest
from datetime import date
from unittest.mock import Mock

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from epymorph.time import Dim, EpiWeek, NBins, epi_week, epi_year_first_day


class EpiWeeksTest(unittest.TestCase):
    def test_first_epi_day(self):
        self.assertEqual(epi_year_first_day(2020), pd.Timestamp(2019, 12, 29))
        self.assertEqual(epi_year_first_day(2021), pd.Timestamp(2021, 1, 3))
        self.assertEqual(epi_year_first_day(2022), pd.Timestamp(2022, 1, 2))
        self.assertEqual(epi_year_first_day(2023), pd.Timestamp(2023, 1, 1))
        self.assertEqual(epi_year_first_day(2024), pd.Timestamp(2023, 12, 31))
        self.assertEqual(epi_year_first_day(2025), pd.Timestamp(2024, 12, 29))

    def test_epi_week(self):
        self.assertEqual(epi_week(date(2021, 1, 1)), EpiWeek(2020, 53))
        self.assertEqual(epi_week(date(2021, 1, 2)), EpiWeek(2020, 53))
        self.assertEqual(epi_week(date(2021, 1, 3)), EpiWeek(2021, 1))
        self.assertEqual(epi_week(date(2024, 1, 1)), EpiWeek(2024, 1))
        self.assertEqual(epi_week(date(2024, 1, 6)), EpiWeek(2024, 1))
        self.assertEqual(epi_week(date(2024, 1, 7)), EpiWeek(2024, 2))
        self.assertEqual(epi_week(date(2024, 3, 14)), EpiWeek(2024, 11))
        self.assertEqual(epi_week(date(2024, 12, 28)), EpiWeek(2024, 52))
        self.assertEqual(epi_week(date(2024, 12, 29)), EpiWeek(2025, 1))
        self.assertEqual(epi_week(date(2024, 12, 31)), EpiWeek(2025, 1))


class TestNBins(unittest.TestCase):
    def _do_test(
        self,
        *,
        bins: int,
        nodes: int,
        days: int,
        tau_steps: int,
        expected_bins: int,
        expected_ticks_per_bin: int,
        tick_offset: int = 0,  # tick indices don't have to start from zero!
    ) -> None:
        dim = Dim(nodes, days, tau_steps)
        t = np.arange(dim.days * dim.tau_steps).repeat(dim.nodes) + tick_offset
        d = Mock(np.array)  # NBins doesn't use dates, so mock is fine
        exp = np.arange(expected_bins).repeat(nodes).repeat(expected_ticks_per_bin)
        act = NBins(bins).map(dim, t, d)
        assert_array_equal(exp, act)

    def test_nbins_01(self):
        # Simple case
        self._do_test(
            bins=10,
            nodes=1,
            days=10,
            tau_steps=3,
            tick_offset=0,
            expected_bins=10,
            expected_ticks_per_bin=3,
        )

    def test_nbins_02(self):
        # Simple case with non-zero tick offset
        self._do_test(
            bins=10,
            nodes=1,
            days=10,
            tau_steps=3,
            tick_offset=3,
            expected_bins=10,
            expected_ticks_per_bin=3,
        )

    def test_nbins_03(self):
        # Simple case with more than one node
        self._do_test(
            bins=10,
            nodes=5,
            days=10,
            tau_steps=3,
            tick_offset=0,
            expected_bins=10,
            expected_ticks_per_bin=3,
        )

    def test_nbins_04(self):
        # We can wind up with more bins than asked for
        self._do_test(
            bins=10,  # not evenly divisible
            nodes=1,
            days=11,
            tau_steps=3,
            tick_offset=0,
            expected_bins=11,
            expected_ticks_per_bin=3,
        )

    def test_nbins_05(self):
        # We can wind up with less bins than asked for
        self._do_test(
            bins=25,  # more than one bin per day
            nodes=2,
            days=10,
            tau_steps=2,
            tick_offset=0,
            expected_bins=10,
            expected_ticks_per_bin=2,
        )
